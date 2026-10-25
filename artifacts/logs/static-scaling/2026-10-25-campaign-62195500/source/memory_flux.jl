"""
Phase 5 Flux memory pilot.

Runs complete training steps at one global batch size on one local shard per
rank, after warm-up, and records device memory counters by phase:

  forward   post-forward live and reserved memory
  backward  post-backward live and reserved memory
  sync      post-allreduce/guard live and reserved memory (W > 1)
  optimizer post-AdamW-update live and reserved memory
  total     maximum of the phase-boundary counters

The governing pass/fail counter is the maximum per-process device memory from
`nvidia-smi`. Allocator live/reserved counters are recorded separately.
Unavailable counters are written as null with an explicit status string.

Reproduction (Leonardo compute node, W = 1, 2, or 4):

  julia --startup-file=no memory_flux.jl \
    --fixture-dir <PHASE5_FIXTURE> --world-size 1 --global-batch 512 \
    --warmup 2 --updates 3 --out-dir <OUT>/memory-b512/flux-w1
"""

include(joinpath(@__DIR__, "replay.jl"))
include(joinpath(@__DIR__, "model_flux.jl"))
include(joinpath(@__DIR__, "precision.jl"))
include(joinpath(@__DIR__, "phase4_common.jl"))
using Flux, CUDA, MPI, NCCL, SHA, Sockets
using .StaticScalingFlux
const P4 = StaticScalingPhase4

function _opt_int(opts, key, default)
    s = get(opts, key, string(default))
    v = tryparse(Int, s)
    v === nothing && error("--$(key) must be an integer, got $(repr(s))")
    return v
end

_mem_value(x) = x === missing ? nothing : Int(x)

function _memory_snapshot()
    live = try _mem_value(CUDA.used_memory()) catch; nothing end
    reserved = try _mem_value(CUDA.cached_memory()) catch; nothing end
    free = try Int(CUDA.free_memory()) catch; nothing end
    total = try Int(CUDA.total_memory()) catch; nothing end
    return (live = live, reserved = reserved, free = free, total = total)
end

function _process_gpu_bytes()
    out = try
        read(`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`, String)
    catch
        return (nothing, "nvidia-smi_unavailable")
    end
    pid = getpid()
    values = Int[]
    for line in split(strip(out), '\n')
        parts = split(line, ',')
        length(parts) >= 2 || continue
        p = tryparse(Int, strip(parts[1]))
        m = tryparse(Int, strip(parts[2]))
        (p === nothing || m === nothing) && continue
        p == pid && push!(values, m * 1024^2)
    end
    isempty(values) && return (nothing, "pid_not_listed")
    return (maximum(values), "measured")
end

mutable struct Peaks
    live::Dict{Symbol,Int}
    reserved::Dict{Symbol,Int}
    process::Dict{Symbol,Int}
    process_status::String
end

Peaks() = Peaks(Dict{Symbol,Int}(), Dict{Symbol,Int}(), Dict{Symbol,Int}(), "measured")

function _record!(peaks::Peaks, phase::Symbol, snap, proc)
    if snap.live !== nothing
        peaks.live[phase] = max(get(peaks.live, phase, 0), snap.live)
    end
    if snap.reserved !== nothing
        peaks.reserved[phase] = max(get(peaks.reserved, phase, 0), snap.reserved)
    end
    bytes, status = proc
    peaks.process_status = status == "measured" ? peaks.process_status :
                           (peaks.process_status == "measured" ? status : peaks.process_status)
    if bytes !== nothing
        peaks.process[phase] = max(get(peaks.process, phase, 0), bytes)
    end
    return nothing
end

_bytes(a) = Int(length(a) * sizeof(eltype(a)))

function _model_bytes(model)
    return sum(_bytes(param_by_name(model, name)) for name in bundle_parameter_names(); init = 0)
end

function _optimizer_state_bytes(state)
    leaves = P4._state_leaf_map(state)
    total = 0
    for (_, leaf) in leaves
        m, v, _ = leaf.state
        m === nothing || (total += _bytes(m))
        v === nothing || (total += _bytes(v))
        total += 2 * sizeof(Float32)
    end
    return total
end

function _batch_bytes(x, y)
    return _bytes(x) + _bytes(y)
end

function _clear_cuda!(refs...)
    GC.gc()
    CUDA.synchronize()
    CUDA.reclaim()
    CUDA.synchronize()
    return nothing
end

function _setup_model(bundle, dev, world, backend)
    model = dev(make_model(bundle))
    state = world == 1 ? build_optimizer_state(model; eta = schedule_eta(1)) :
            P4.build_distributed_state(model, backend; eta = schedule_eta(1))
    return model, state
end

function _train_step_public!(model, state, x, y, eta)
    return train_step!(model, state, x, y; eta = eta)
end

function _train_step_measured!(model, state, backend, x, y, eta, peaks)
    forward_snapshot = Ref{Any}(nothing)
    val, grads = Zygote.withgradient(model) do m
        logits, _ = forward_with_activations(m, x; capture = false)
        loss = Flux.Losses.logitcrossentropy(logits, y)
        CUDA.synchronize()
        forward_snapshot[] = _memory_snapshot()
        return (loss = loss, logits = logits)
    end
    CUDA.synchronize()
    _record!(peaks, :forward, forward_snapshot[], _process_gpu_bytes())
    _record!(peaks, :backward, _memory_snapshot(), _process_gpu_bytes())

    g = grads[1]
    if backend === nothing
        Optimisers.adjust!(state, eta)
        state, model = Optimisers.update!(state, model, g)
    else
        synced = Flux.DistributedUtils._sync_gradients(backend, state.tree, model, g)
        CUDA.synchronize()
        _record!(peaks, :sync, _memory_snapshot(), _process_gpu_bytes())
        Optimisers.adjust!(state.tree, eta)
        tree2, model2 = Optimisers.update!(state.tree, model, synced)
        state = Flux.DistributedUtils.DistributedOptimizerState(state.backend, tree2)
        model = model2
    end
    CUDA.synchronize()
    _record!(peaks, :optimizer, _memory_snapshot(), _process_gpu_bytes())
    return model, state, val.loss, val.logits
end

function memory_main()
    opts = parse_cli_args(ARGS)
    world = _opt_int(opts, "world-size", 1)
    world in (1, 2, 4) || error("--world-size must be 1, 2, or 4")
    global_batch = _opt_int(opts, "global-batch", 512)
    warmup = _opt_int(opts, "warmup", 2)
    updates = _opt_int(opts, "updates", 3)
    global_batch in (512, 256) || error("--global-batch must be 512 or 256")
    global_batch % world == 0 || error("--global-batch $(global_batch) not divisible by world $(world)")
    warmup >= 1 || error("--warmup must be >= 1")
    updates >= 1 || error("--updates must be >= 1")
    local_batch = global_batch ÷ world
    out_dir = abspath(get(opts, "out-dir", ""))
    isempty(out_dir) && error("--out-dir is required")
    fixture_dir = abspath(get(opts, "fixture-dir", ""))
    isempty(fixture_dir) && error("--fixture-dir is required")
    data_path = joinpath(fixture_dir, "data_fixture.npz")
    bundle_path = joinpath(fixture_dir, "initial_bundle.npz")
    isfile(data_path) || error("missing $(data_path)")
    isfile(bundle_path) || error("missing $(bundle_path)")

    backend = nothing
    rank = 0
    if world > 1
        Flux.DistributedUtils.initialize(Flux.NCCLBackend)
        backend = Flux.DistributedUtils.get_distributed_backend(Flux.NCCLBackend)
        rank = Flux.DistributedUtils.local_rank(backend)
        @assert Flux.DistributedUtils.total_workers(backend) == world
    end
    P4._configure_precision!()
    CUDA.allowscalar(false)
    world == 1 && CUDA.device!(0)
    dev = Flux.gpu_device(; force = true)
    device_index = Int(CUDA.deviceid(CUDA.device()))

    data = read_npz_fields(data_path)
    bundle = read_npz_fields(bundle_path)
    refs = data["train_transformed_ref"]
    labels = vec(Int64.(data["train_labels"]))
    size(refs, 2) == global_batch ||
        error("fixture batch $(size(refs, 2)) != --global-batch $(global_batch)")
    size(labels, 1) == global_batch ||
        error("fixture label count $(size(labels, 1)) != --global-batch $(global_batch)")

    batches = map(1:2) do row
        x, y = P4._global_batch(data, row)
        x, y, _ = P4.shard_batch(x, y, rank + 1, world)
        P4._to_training_batch(x, y, dev)
    end

    model, state = _setup_model(bundle, dev, world, backend)
    for t in 1:warmup
        x, y = batches[mod1(t, 2)]
        model, state, _ = _train_step_public!(model, state, x, y, schedule_eta(t))
    end
    CUDA.synchronize()

    model = nothing
    state = nothing
    _clear_cuda!()
    model, state = _setup_model(bundle, dev, world, backend)
    CUDA.synchronize()
    _clear_cuda!()

    baseline = _memory_snapshot()
    baseline_proc, baseline_status = _process_gpu_bytes()
    peaks = Peaks()
    peaks.process_status = baseline_status
    _record!(peaks, :baseline, baseline, (baseline_proc, baseline_status))

    for t in 1:updates
        x, y = batches[mod1(t, 2)]
        model, state, loss, _ = _train_step_measured!(
            model, state, backend, x, y, schedule_eta(t), peaks)
        @assert isfinite(P4.loss32(loss))
    end
    CUDA.synchronize()
    final_snapshot = _memory_snapshot()
    final_proc, final_status = _process_gpu_bytes()
    _record!(peaks, :final, final_snapshot, (final_proc, final_status))

    live_peak = isempty(values(peaks.live)) ? nothing : maximum(values(peaks.live))
    reserved_peak = isempty(values(peaks.reserved)) ? nothing : maximum(values(peaks.reserved))
    process_peak = isempty(values(peaks.process)) ? nothing : maximum(values(peaks.process))
    total = baseline.total === nothing ? final_snapshot.total : baseline.total
    margin_limit = total === nothing ? nothing : floor(Int, 0.8 * total)
    required_free = total === nothing ? nothing : ceil(Int, 0.2 * total)

    record = Dict{String,Any}(
        "schema" => "static-memory-pilot-v1",
        "framework" => "flux",
        "world_size" => world,
        "global_batch" => global_batch,
        "local_batch" => local_batch,
        "warmup" => warmup,
        "updates" => updates,
        "rank" => rank,
        "device" => device_index,
        "node" => gethostname(),
        "job" => get(ENV, "SLURM_JOB_ID", ""),
        "fixture_sha256" => bytes2hex(sha256(read(data_path))),
        "bundle_sha256" => bytes2hex(sha256(read(bundle_path))),
        "workload" => "two-real-global-batches-from-phase5-fixture",
        "precision" => "FP32 strict FMA, TF32 off",
        "staging_bytes" => sum(_batch_bytes(x, y) for (x, y) in batches),
        "model_bytes" => _model_bytes(model),
        "optimizer_state_bytes" => _optimizer_state_bytes(state),
        "device_total_bytes" => total,
        "device_free_baseline_bytes" => baseline.free,
        "device_process_baseline_bytes" => baseline_proc,
        "device_process_peak_bytes" => process_peak,
        "allocator_live_baseline_bytes" => baseline.live,
        "allocator_live_peak_bytes" => live_peak,
        "allocator_reserved_baseline_bytes" => baseline.reserved,
        "allocator_reserved_peak_bytes" => reserved_peak,
        "phases" => Dict(
            "forward_live_bytes" => get(peaks.live, :forward, nothing),
            "forward_reserved_bytes" => get(peaks.reserved, :forward, nothing),
            "backward_live_bytes" => get(peaks.live, :backward, nothing),
            "backward_reserved_bytes" => get(peaks.reserved, :backward, nothing),
            "sync_live_bytes" => get(peaks.live, :sync, nothing),
            "sync_reserved_bytes" => get(peaks.reserved, :sync, nothing),
            "optimizer_live_bytes" => get(peaks.live, :optimizer, nothing),
            "optimizer_reserved_bytes" => get(peaks.reserved, :optimizer, nothing),
        ),
        "counter_status" => Dict(
            "allocator_live" => "measured_at_phase_boundaries",
            "allocator_reserved" => "high_water_mark_after_reclaim",
            "device_process" => peaks.process_status,
            "communication" => world > 1 ? "measured_sync_phase" : "not_applicable",
            "guard" => "host_float32_flags_only; no device allocation",
            "forward_peak" => "phase_boundary",
            "backward_peak" => "phase_boundary",
        ),
        "margin" => Dict(
            "required_free_fraction" => 0.2,
            "limit_used_fraction" => 0.8,
            "limit_bytes" => margin_limit,
            "required_free_bytes" => required_free,
        ),
        "pass_checks" => true,
    )
    rank_path = joinpath(out_dir, @sprintf("rank%d.json", rank))
    P4.write_json(rank_path, record)
    if world > 1
        P4.phase4_barrier(MPI, backend)
    end
    if rank == 0
        rank_files = String[]
        for r in 0:(world - 1)
            p = joinpath(out_dir, @sprintf("rank%d.json", r))
            isfile(p) || error("missing rank record $(p)")
            push!(rank_files, basename(p))
        end
        summary = Dict{String,Any}(
            "schema" => "static-memory-pilot-summary-v1",
            "framework" => "flux",
            "world_size" => world,
            "global_batch" => global_batch,
            "local_batch" => local_batch,
            "warmup" => warmup,
            "updates" => updates,
            "node" => gethostname(),
            "job" => get(ENV, "SLURM_JOB_ID", ""),
            "rank_files" => rank_files,
            "pass_checks" => true,
        )
        P4.write_json(joinpath(out_dir, "summary.json"), summary)
        println("MEMORY_PASS flux world=$(world) batch=$(global_batch)")
    end
    if world > 1
        MPI.Barrier(P4.backend_comm(backend))
    end
    return nothing
end

memory_main()

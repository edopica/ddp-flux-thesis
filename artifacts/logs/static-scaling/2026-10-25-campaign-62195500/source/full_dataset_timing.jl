"""
Full-dataset timing path for the static-scaling Flux benchmark.

Included after `replay.jl`, `model_flux.jl`, `precision.jl`,
`phase4_common.jl`, and `full_dataset.jl`. The fixture timing path in
`timing_flux.jl` is unchanged.
"""

using SHA
using Sockets

function _expected_beta_power(updates::Integer)
    b1 = Float32(0.9)
    b2 = Float32(0.999)
    for _ in 1:updates
        b1 *= Float32(0.9)
        b2 *= Float32(0.999)
    end
    return (b1, b2)
end

function _timing_bool(value)
    return lowercase(strip(String(value))) in ("true", "1", "yes")
end

function _gather_hash_values(mpi_mod, comm, hash_value::AbstractString, world::Integer)
    world == 1 && return [String(hash_value)]
    bytes = Vector{UInt8}(codeunits(hash_value))
    gathered = mpi_mod.Allgather(bytes, comm)
    width = length(bytes)
    length(gathered) == width * world ||
        error("hash gather returned $(length(gathered)) bytes for $(width) x $(world)")
    values = [String(gathered[(rank * width + 1):((rank + 1) * width)])
              for rank in 0:(world - 1)]
    all(==(values[1]), values) ||
        error("final $(hash_value) differs between ranks")
    return values
end

"""
    run_full_dataset_timing(opts; world, updates, warmup, snapshot_dir,
                            api, api_name, backend, rank, dev)

Run end-to-end full-dataset updates and write one rank-0 JSON result with
schema `static-timing-full-dataset-v1`.
"""
function run_full_dataset_timing(opts::AbstractDict;
                                 world = nothing,
                                 updates = nothing,
                                 warmup = nothing,
                                 snapshot_dir = nothing,
                                 api = nothing,
                                 api_name = nothing,
                                 backend = nothing,
                                 rank = nothing,
                                 dev = nothing)
    dataset_dir = get(opts, "dataset-dir", "")
    isempty(dataset_dir) && error("--dataset-dir is required")
    dataset_dir = abspath(dataset_dir)
    epoch = parse(Int, get(opts, "epoch", "0"))
    batch_size = parse(Int, get(opts, "batch-size", "512"))
    batch_size in (512, 256) || error("--batch-size must be 512 or 256; got $(batch_size)")
    expected_updates = batch_size == 512 ? 195 : 390
    updates === nothing && (updates = parse(Int, get(opts, "updates", "195")))
    warmup === nothing && (warmup = parse(Int, get(opts, "warmup", "20")))
    updates = Int(updates)
    warmup = Int(warmup)
    world === nothing && (world = parse(Int, opts["world-size"]))
    world = Int(world)
    0 <= epoch <= 49 || error("--epoch must be in 0:49; got $(epoch)")
    world in (1, 2, 4) || error("--world-size must be 1, 2, or 4; got $(world)")
    1 <= updates <= expected_updates ||
        error("--updates must be in 1:$(expected_updates); got $(updates)")
    2 <= warmup <= 100 || error("--warmup must be in 2:100; got $(warmup)")
    batch_size % world == 0 ||
        error("batch size $(batch_size) is not divisible by world size $(world)")

    bundle_path = get(opts, "bundle", "")
    isempty(bundle_path) && (bundle_path = joinpath(dataset_dir, "initial_bundle.npz"))
    bundle_path = abspath(bundle_path)
    isfile(bundle_path) || error("bundle not found: $(bundle_path)")

    out_path = get(opts, "out", "")
    isempty(out_path) && error("--out is required")
    snapshot_dir === nothing && (snapshot_dir = get(opts, "snapshot-dir", ""))
    snapshot_dir = String(snapshot_dir)
    verify_large_hashes = _timing_bool(get(opts, "verify-large-hashes", "false"))

    if api === nothing
        name = api_name === nothing ? lowercase(get(opts, "distributed-api", "guarded")) :
               lowercase(String(api_name))
        name in ("guarded", "static") ||
            error("--distributed-api must be guarded or static; got $(repr(name))")
        api_name = name
        api = Symbol(name)
    else
        api in (:guarded, :static) ||
            error("distributed api must be :guarded or :static; got $(repr(api))")
        api_name = String(api)
    end

    if world > 1
        if backend === nothing
            Flux.DistributedUtils.initialize(Flux.NCCLBackend)
            backend = Flux.DistributedUtils.get_distributed_backend(Flux.NCCLBackend)
        end
        rank === nothing && (rank = Flux.DistributedUtils.local_rank(backend))
        @assert Flux.DistributedUtils.total_workers(backend) == world
        rank = Int(rank)
    else
        backend === nothing || error("world size 1 does not use a distributed backend")
        rank === nothing && (rank = 0)
        rank == 0 || error("world size 1 requires rank 0")
    end

    P4._configure_precision!()
    CUDA.allowscalar(false)
    dev === nothing && (dev = Flux.gpu_device(; force = true))
    if haskey(ENV, "FULL_PREFLIGHT_RANK_DIR")
        # Each rank owns its own file; this is outside the measured region.
        open(joinpath(ENV["FULL_PREFLIGHT_RANK_DIR"], "rank-$(rank).json"), "w") do io
            print(io, "{\"rank\":", rank, ",\"pid\":", getpid(),
                  ",\"gpu_uuid\":\"GPU-", CUDA.uuid(CUDA.device()), "\"}")
        end
    end
    full_eta(t) = schedule_eta(t; W = 5 * expected_updates, T = 50 * expected_updates)

    ds = open_full_dataset(dataset_dir; verify_large_hashes = verify_large_hashes)
    size(ds.train_images) == (64, 64, 3, 100000) ||
        error("full-dataset timing requires 100000 training images; got $(size(ds.train_images))")
    size(ds.train_labels) == (100000,) ||
        error("full-dataset timing requires 100000 labels; got $(size(ds.train_labels))")
    size(ds.permutations) == (100000, 50) ||
        error("full-dataset timing requires a (100000, 50) permutation table; got $(size(ds.permutations))")
    size(ds.augmentation) == (3, 100000, 50) ||
        error("full-dataset timing requires a (3, 100000, 50) augmentation table; got $(size(ds.augmentation))")
    bundle = read_npz_fields(bundle_path)
    bundle_sha256 = _sha256_file(bundle_path)
    n_eff = length(epoch_ids(ds, epoch, batch_size))
    n_updates_epoch = n_eff ÷ batch_size
    n_updates_epoch == expected_updates ||
        error("dataset epoch $(epoch) has $(n_updates_epoch) updates at batch $(batch_size); expected $(expected_updates)")

    function setup()
        model = dev(make_model(bundle))
        state = world == 1 ? build_optimizer_state(model; eta = full_eta(1)) :
                P4.build_distributed_state(model, backend; eta = full_eta(1), api = api)
        return model, state
    end

    function steps(model, state, count)
        loss = 0f0
        for t in 1:count
            x_whcn, labels, _, _ = local_batch(ds, epoch, batch_size, t, rank + 1, world)
            x = dev(x_whcn)
            y = dev(Flux.onehotbatch(labels .+ 1, 1:200))
            res = train_step!(model, state, x, y; eta = full_eta(t))
            model, state, loss = res.model, res.opt_state, res.loss
        end
        return model, state, loss
    end

    model, state = setup()
    steps(model, state, warmup)
    CUDA.synchronize()
    model, state = setup()
    GC.gc()
    CUDA.synchronize()
    P4.phase4_barrier(MPI, backend)
    CUDA.synchronize()

    # Wall-clock epoch timestamps bracket the same synchronized boundaries as the
    # monotonic timing measurement so the external nvidia-smi sampler can be
    # clipped to the post-warmup training window. Final hashing, snapshots, and
    # teardown stay outside this window.
    start_epoch = time()
    start = time_ns()
    model, state, loss = steps(model, state, updates)
    CUDA.synchronize()
    end_epoch = time()
    elapsed = (time_ns() - start) / 1e9
    @assert isfinite(P4.loss32(loss))

    expected_beta = _expected_beta_power(updates)
    for (name, leaf) in P4._state_leaf_map(state)
        leaf.state[3] == expected_beta ||
            error("optimizer leaf $(name) beta_power $(leaf.state[3]) does not match $(updates) timed updates")
    end
    eta_values = Float32[Float32(full_eta(t)) for t in 1:updates]
    eta_sha256 = bytes2hex(SHA.sha256(reinterpret(UInt8, eta_values)))

    hashes = P4.state_hashes(model, state)
    comm = world == 1 ? nothing : P4.backend_comm(backend)
    if world > 1
        for key in ("param", "m", "v", "beta_power")
            P4.broadcast_hash(MPI, backend, hashes[key], "final " * key)
        end
    end
    rank_hashes = Dict(key => _gather_hash_values(MPI, comm, hashes[key], world)
                       for key in keys(hashes))
    durations = world == 1 ? [elapsed] : MPI.Allgather(elapsed, comm)
    window_starts = world == 1 ? [start_epoch] : MPI.Allgather(start_epoch, comm)
    window_ends = world == 1 ? [end_epoch] : MPI.Allgather(end_epoch, comm)
    rank_training_windows = [
        Dict("rank" => index - 1, "start_epoch" => window_starts[index],
             "end_epoch" => window_ends[index], "seconds" => durations[index])
        for index in eachindex(durations)
    ]

    if !isempty(snapshot_dir) && rank == 0
        mkpath(snapshot_dir)
        names = P4.bundle_parameter_names()
        leaves = P4._state_leaf_map(state)
        entries = Pair{String,Any}[]
        for name in names
            leaf = leaves[name]
            push!(entries, "param.$(name)" => P4.canonical_param(name, P4.host_array(P4.param_by_name(model, name))))
            push!(entries, "m.$(name)" => P4.canonical_param(name, P4.host_array(leaf.state[1])))
            push!(entries, "v.$(name)" => P4.canonical_param(name, P4.host_array(leaf.state[2])))
            bp = leaf.state[3]
            push!(entries, "beta_power.$(name)" => Float32[Float32(bp[1]), Float32(bp[2])])
        end
        push!(entries, "step" => Int64[updates])
        P4._write_npz(joinpath(snapshot_dir, "state.npz"), entries)
    end

    if rank == 0
        seconds = maximum(durations)
        source = P4.source_identity(opts)
        config_kind = world == 1 ? "single_gpu" :
                      (api_name == "static" ? "historical_static_ddp" : "guarded_ddp")
        guard_policy = world == 1 ? "not applicable: single-GPU baseline" :
                       (api_name == "static" ?
                        "no presence or nonfinite distributed validation" :
                        "presence + nonfinite loss/gradient rejection inside timed window")
        host_maxrss_bytes = Int64(Sys.maxrss())
        free_bytes, total_bytes = CUDA.memory_info()
        memory = Dict{String,Any}(
            "host_maxrss_bytes" => host_maxrss_bytes,
            "device_free_bytes" => Int64(free_bytes),
            "device_total_bytes" => Int64(total_bytes),
            "device_used_bytes" => Int64(total_bytes) - Int64(free_bytes),
        )
        payload = Dict{String,Any}(
            "schema" => "static-timing-full-dataset-v1",
            "framework" => "flux",
            "world_size" => world,
            "global_batch" => batch_size,
            "batch_size" => batch_size,
            "epoch" => epoch,
            "updates" => updates,
            "warmup" => warmup,
            "updates_per_epoch" => expected_updates,
            "n_eff" => n_eff,
            "tail_size" => size(ds.permutations, 1) - n_eff,
            "seconds" => seconds,
            "images_per_second" => batch_size * updates / seconds,
            "update_seconds" => seconds / updates,
            "rank_seconds" => durations,
            "rank_training_windows" => rank_training_windows,
            "training_window_basis" => "per-rank wall-clock epoch seconds recorded at the synchronized start and after the synchronized end of the measured updates; warmup, reset, final hashing, snapshots, and teardown are outside the window",
            "hashes" => hashes,
            "rank_hashes" => rank_hashes,
            "device" => Int(CUDA.deviceid(CUDA.device())),
            "node" => gethostname(),
            "job" => get(ENV, "SLURM_JOB_ID", ""),
            "dataset_dir" => ds.dataset_dir,
            "dataset_hashes" => ds.metadata["files"],
            "bundle_path" => bundle_path,
            "bundle_sha256" => bundle_sha256,
            "bundle_file_sha256" => bundle_sha256,
            "bundle_hash" => ds.metadata["declared_bundle_hash"],
            "sample_order_sha256" => sample_order_sha256(ds, epoch, batch_size),
            "update1_order_sha256" => update_order_sha256(ds, epoch, batch_size, 1),
            "transform_spec_sha256" => transform_spec_sha256(ds),
            "loader_policy" => "mmap opened once per process before timing; every timed update assembles its batch from the map",
            "timing" => "full-dataset-end-to-end",
            "timing_policy" => "mmap opened before timing; timed region includes batch assembly, transform, onehot, H2D, forward, backward, and optimizer step",
            "warmup_policy" => "$(warmup) discarded updates at t=1..$(warmup); model and optimizer reset from the bundle; measured region restarts at update 1 and data at batch 1",
            "workload" => "full-dataset-epoch",
            "precision" => "FP32 strict FMA, TF32 off",
            "snapshot" => !isempty(snapshot_dir),
            "pass_checks" => true,
            "distributed_api" => world == 1 ? "single_gpu" : api_name,
            "config_kind" => config_kind,
            "guard_policy" => guard_policy,
            "source" => source,
            "flux_repo_root" => source["flux_repo_root"],
            "flux_commit" => source["flux_commit"],
            "flux_state_sha256" => source["flux_state_sha256"],
            "memory" => memory,
            "eta_sha256" => eta_sha256,
            "final_loss" => P4.loss32(loss),
            "loss_finite" => true,
        )
        P4.write_json(out_path, payload)
        println("TIMING_PASS flux-full $(world) $(seconds)")
    end
    return nothing
end

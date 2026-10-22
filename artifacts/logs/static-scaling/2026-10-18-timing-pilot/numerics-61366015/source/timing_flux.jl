include(joinpath(@__DIR__, "replay.jl"))
include(joinpath(@__DIR__, "model_flux.jl"))
include(joinpath(@__DIR__, "precision.jl"))
include(joinpath(@__DIR__, "phase4_common.jl"))
using Flux, CUDA, MPI, NCCL, SHA, Sockets
using .StaticScalingFlux
const P4 = StaticScalingPhase4

function timing_main()
    opts = parse_cli_args(ARGS)
    world = parse(Int, opts["world-size"])
    updates = parse(Int, get(opts, "updates", "300"))
    warmup = parse(Int, get(opts, "warmup", "20"))
    snapshot_dir = get(opts, "snapshot-dir", "")
    @assert world in (1, 2, 4) && 1 <= updates <= 9750 && 2 <= warmup <= 100
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
    dev = Flux.gpu_device(; force=true)
    fixture = joinpath(opts["fixture-dir"], "data_fixture.npz")
    bundle_path = joinpath(opts["fixture-dir"], "initial_bundle.npz")
    data = read_npz_fields(fixture)
    bundle = read_npz_fields(bundle_path)
    batches = map(1:2) do row
        x, y = P4._global_batch(data, row)
        x, y, _ = P4.shard_batch(x, y, rank + 1, world)
        P4._to_training_batch(x, y, dev)
    end
    function setup()
        model = dev(make_model(bundle))
        state = world == 1 ? build_optimizer_state(model; eta=schedule_eta(1)) :
            P4.build_distributed_state(model, backend; eta=schedule_eta(1))
        model, state
    end
    function steps(model, state, count)
        loss = 0f0
        for t in 1:count
            x, y = batches[mod1(t, 2)]
            res = train_step!(model, state, x, y; eta=schedule_eta(t))
            model, state, loss = res.model, res.opt_state, res.loss
        end
        model, state, loss
    end
    model, state = setup()
    steps(model, state, warmup)
    CUDA.synchronize()
    model, state = setup()
    GC.gc()
    CUDA.synchronize()
    P4.phase4_barrier(MPI, backend)
    CUDA.synchronize()
    start = time_ns()
    model, state, loss = steps(model, state, updates)
    CUDA.synchronize()
    elapsed = (time_ns() - start) / 1e9
    @assert isfinite(P4.loss32(loss))
    hashes = P4.state_hashes(model, state)
    if world > 1
        for key in ("param", "m", "v", "beta_power")
            P4.broadcast_hash(MPI, backend, hashes[key], "final " * key)
        end
    end
    durations = world == 1 ? [elapsed] : MPI.Allgather(elapsed, P4.backend_comm(backend))
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
        P4._write_npz(joinpath(snapshot_dir, "state.npz"), entries)
    end
    if rank == 0
        seconds = maximum(durations)
        P4.write_json(opts["out"], Dict("schema"=>"static-timing-pilot-v1", "framework"=>"flux",
            "world_size"=>world, "global_batch"=>12, "updates"=>updates, "warmup"=>warmup,
            "seconds"=>seconds, "images_per_second"=>12*updates/seconds,
            "update_seconds"=>seconds/updates, "rank_seconds"=>durations,
            "hashes"=>hashes, "node"=>gethostname(), "job"=>get(ENV, "SLURM_JOB_ID", ""),
            "fixture_sha256"=>bytes2hex(sha256(read(fixture))),
            "bundle_sha256"=>bytes2hex(sha256(read(bundle_path))),
            "workload"=>"alternating-two-fixture-batches", "timing"=>"training-only",
            "precision"=>"FP32 strict FMA, TF32 off", "snapshot"=>!isempty(snapshot_dir),
            "pass_checks"=>true))
        println("TIMING_PASS flux $(world) $(seconds)")
    end
end
timing_main()

#!/usr/bin/env julia
# Phase K post-completion timing stretch CLI.
#
# Modes:
#   single  one process; evaluates every scheduled shard of the saved global
#           schedule and applies one native Optimisers update per step.
#   ddp     MPI/NCCL job; one local shard per rank and the distributed wrapper
#           (`DistributedOptimizerState`) with the real backend.
#
# Both modes run the same fixed work (one saved schedule, `--steps` global
# updates), warm up untimed, rebuild the initial state from the prepared
# artifact before each timed repeat, and validate only at record boundaries.
# Timed and warm-up steps use the plain native update and take no host
# snapshots; only `--check-trajectory` keeps the assertion-heavy step. The
# barrier-guarded diagnostic window excludes validation and record writes from
# the training-only time, so single and DDP train intervals are comparable.
# Ratios of medians belong to the driver and plotter, not to this script.
# The timed region, console markers, and record schemas follow
# `temp/phase_k_timing_stretch_plan.md`.
#
# Run under the Flux implementation environment, for example:
#   julia --project=../flux-integration/test scripts/examples/conditional_graph_timing.jl \
#       single --prepared <p.jls> --out <dir> --device cpu --world 4
#   mpiexec -n 4 julia --project=../flux-integration/test \
#       scripts/examples/conditional_graph_timing.jl ddp --prepared <p.jls> --out <dir> --backend mpi

import Serialization: serialize, deserialize

const _RAW_ARGS = copy(ARGS)

function _raw_option(args, key, default=nothing)
    i = findfirst(==("--" * key), args)
    i === nothing && return default
    i < length(args) || error("option --$key requires a value")
    return args[i + 1]
end

const MODE = isempty(_RAW_ARGS) ?
    error("usage: conditional_graph_timing.jl single|ddp --prepared <p.jls> --out <dir> [options]") :
    _RAW_ARGS[1]
const BACKEND_ARG = _raw_option(_RAW_ARGS, "backend", "mpi")
const DEVICE_ARG = _raw_option(_RAW_ARGS, "device", "cpu")

# GPU stacks load only for an explicit NCCL backend or a CUDA single run, so a
# CPU single process never touches CUDA. Device arrays are mapped only after
# `CUDA.allowscalar(false)` is set, before model construction.
if BACKEND_ARG == "nccl"
    import CUDA
    import NCCL
    import cuDNN
elseif DEVICE_ARG == "cuda"
    import CUDA
end

using MPI, Flux, Optimisers, Zygote, Functors, Random, Printf, Dates, Serialization
using Flux: DistributedUtils

const DU = DistributedUtils
const W = DistributedUtils.DistributedOptimizerState

include(joinpath(@__DIR__, "conditional_graph", "common.jl"))
include(joinpath(@__DIR__, "conditional_graph", "model.jl"))
include(joinpath(@__DIR__, "conditional_graph", "dataset.jl"))
include(joinpath(@__DIR__, "conditional_graph", "schedule.jl"))
include(joinpath(@__DIR__, "conditional_graph", "reference.jl"))

using .CGCommon, .CGModel, .CGDataset, .CGSchedule, .CGReference

const RECORD_COLUMNS = split("run_id mode repeat step elapsed_s train_s samples_total samples_a samples_b updates_per_s samples_per_s val_a_loss val_a_acc val_b_loss val_b_acc")
const STEP_COLUMNS = split("mode repeat step elapsed_s train_s samples_total samples_a samples_b")
const REPEAT_COLUMNS = split("run_id mode repeat t_total_s t_train_s t_val_s t_write_s slowest_rank_s slowest_train_s steps samples_total updates_per_s samples_per_s train_updates_per_s train_samples_per_s")

const BOOLEAN_OPTIONS = Set(["help"])

function parse_options(args)
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 2
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected positional argument $(repr(a))")
        key = a[3:end]
        if key in BOOLEAN_OPTIONS
            push!(flags, key)
            i += 1
        else
            i + 1 <= length(args) || error("option --$key requires a value")
            opts[key] = args[i + 1]
            i += 2
        end
    end
    return opts, flags
end

require_opt(opts, key) = haskey(opts, key) ? opts[key] :
                         error("missing required option --$key")
int_opt(opts, key, default) = haskey(opts, key) ? parse(Int, opts[key]) : default
float_opt(opts, key, default) = haskey(opts, key) ? parse(Float64, opts[key]) : default

const ON_GPU = Ref(false)
const TARGET_DTYPE = Ref{DataType}(Float32)

is_on_gpu() = ON_GPU[]

# ---------------------------------------------------------------------------
# Device-generic conversion helpers (duplicated from the demo CLI on purpose:
# the timing path must stay independent of the demo's diagnostic code).
# ---------------------------------------------------------------------------

function to_target(x::AbstractArray{<:Real})
    return is_on_gpu() ? CUDA.cu(Float32.(x)) : TARGET_DTYPE[].(x)
end

convert_array(x::AbstractArray) = to_target(x)
convert_array(x) = x

to_target_model(m) = Functors.fmap(convert_array, m)
to_target_tree(t) = Functors.fmap(convert_array, t)

# ---------------------------------------------------------------------------
# Backend initialization (real backend, no recording wrapper)
# ---------------------------------------------------------------------------

function init_backend(backend_name::AbstractString)
    if backend_name == "nccl"
        DU.initialize(DU.NCCLBackend)
        backend = DU.get_distributed_backend(DU.NCCLBackend)
        CUDA.allowscalar(false)
        CUDA.functional() || error("NCCL backend requested but CUDA is not functional")
    elseif backend_name == "mpi"
        DU.initialize(DU.MPIBackend; cuda_devices = missing, amdgpu_devices = missing)
        backend = DU.get_distributed_backend(DU.MPIBackend)
    else
        error("unknown backend $(repr(backend_name)); expected \"mpi\" or \"nccl\"")
    end
    rank = DU.local_rank(backend)
    workers = DU.total_workers(backend)
    comm = CGCommon.diagnostic_comm(backend)
    return (; backend, rank, workers, comm)
end

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

function timing_source_hashes()
    demo_dir = joinpath(@__DIR__, "conditional_graph")
    files = Dict{String,String}()
    for f in ("common.jl", "model.jl", "dataset.jl", "schedule.jl",
        "reference.jl", "checkpoint.jl", "metrics.jl")
        files["scripts/examples/conditional_graph/$f"] =
            CGCommon.source_hash(joinpath(demo_dir, f))
    end
    files["scripts/examples/conditional_graph_timing.jl"] = CGCommon.source_hash(@__FILE__)
    checker = normpath(joinpath(@__DIR__, "..", "checks", "conditional_graph_timing_check.jl"))
    isfile(checker) &&
        (files["scripts/checks/conditional_graph_timing_check.jl"] = CGCommon.source_hash(checker))
    flux_root = normpath(joinpath(dirname(pathof(Flux)), ".."))
    for rel in ("src/Flux.jl", "src/distributed/public_api.jl")
        p = joinpath(flux_root, rel)
        isfile(p) && (files["flux-integration/$rel"] = CGCommon.source_hash(p))
    end
    return files
end

function metadata_text(; mode, label, backend, world, rank, dtype, tol, sched, dataset,
        prepared_path, prepared_sha, run_id, steps, warmup, repeats, record_every,
        eta, device_rows, extra::NamedTuple)
    io = IOBuffer()
    println(io, "mode\t", mode)
    println(io, "label\t", label)
    println(io, "date\t", Dates.now())
    println(io, "host\t", CGCommon.host_name())
    println(io, "julia\t", VERSION)
    println(io, "backend\t", backend)
    println(io, "workers\t", world)
    println(io, "rank\t", rank)
    println(io, "dtype\t", dtype)
    println(io, "command\t", Base.julia_cmd(), " ", join(_RAW_ARGS, ' '))
    println(io, "args\t", join(_RAW_ARGS, ' '))
    println(io, "run_id\t", run_id)
    println(io, "steps\t", steps)
    println(io, "warmup\t", warmup)
    println(io, "repeats\t", repeats)
    println(io, "record_every\t", record_every)
    for (r, device) in device_rows
        println(io, "device\t$r\t", device)
    end
    println(io, "tolerances\t", tol)
    println(io, "eta\t", eta)
    println(io, "prepared\t", abspath(prepared_path))
    println(io, "prepared_sha256\t", prepared_sha)
    println(io, "dataset\t$(dataset.name)\tdigest=$(dataset.digest)" *
                "\ttrain_per_task=$(dataset.train_per_task)\tval_per_task=$(dataset.val_per_task)")
    println(io, "schedule\tdigest=$(CGSchedule.schedule_digest(sched))" *
                "\tsteps=$(sched.steps)\tglobal_batch=$(sched.global_batch)" *
                "\tlocal_batch=$(sched.local_batch)\tworkers=$(sched.workers)")
    println(io, "flux_path\t", pathof(Flux))
    for (name, root) in (("thesis", normpath(joinpath(@__DIR__, "../.."))),
            ("Flux", normpath(joinpath(dirname(pathof(Flux)), ".."))))
        println(io, "git_head\t$name\t", CGCommon.git_head(root))
        for args in (("branch", "--show-current"), ("status", "--short"))
            result = try
                read(`git -C $root $args`, String)
            catch
                "unavailable"
            end
            println(io, "git_$(first(args))\t$name\t", repr(result))
        end
    end
    for name in ("SLURM_JOB_ID", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES")
        println(io, "slurm_env\t$name\t", get(ENV, name, "unset"))
    end
    for pkg in CGCommon.package_versions()
        println(io, "package\t$(pkg.name)\t$(pkg.version)\t$(pkg.path)")
    end
    println(io, "package\tZygote\t$(pkgversion(Zygote))\t$(pathof(Zygote))")
    for (k, v) in sort(collect(pairs(extra)); by = first)
        println(io, "extra\t$k\t$v")
    end
    for (f, h) in sort(collect(timing_source_hashes()); by = first)
        println(io, "source_hash\t$f\t$h")
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

function write_header(path, columns)
    open(path, "w") do io
        println(io, join(columns, '\t'))
    end
end

function append_flush(path, fields)
    open(path, "a") do io
        println(io, join(fields, '\t'))
        flush(io)
    end
end

function median_of(values)
    sorted = sort(values)
    n = length(sorted)
    return isodd(n) ? sorted[(n + 1) ÷ 2] :
           (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

# ---------------------------------------------------------------------------
# Validation (rank 0 only; lean, chunked, tasks reported separately)
# ---------------------------------------------------------------------------

function validate_task(model, dataset, task::Symbol)
    pool = dataset.pools[task].val
    n = size(pool.y, 2)
    total_loss = 0.0
    correct = 0
    for first in 1:128:n
        ids = first:min(first+127, n)
        x = to_target(pool.x[:, ids])
        y = to_target(pool.y[:, ids])
        logits = model(x, task)
        loss = Flux.Losses.logitcrossentropy(logits, y)
        loss isa Number || error("validation loss is not a host scalar")
        total_loss += Float64(loss) * length(ids)
        correct += sum(argmax(Array(logits); dims=1) .== argmax(pool.y[:, ids]; dims=1))
    end
    return (; loss=total_loss/n, accuracy=correct/n, n)
end

function validate_model(model, dataset)
    return (; a=validate_task(model, dataset, :a), b=validate_task(model, dataset, :b))
end

# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

# single: untimed trajectory-check step. One process evaluates all `world`
# scheduled shards of the global objective and applies one native update.
# `reference_step!` snapshots the globally absent head parameters and both Adam
# moments to the host before/after every update; those assertions are only paid
# by `--check-trajectory` and never by a timed repeat.
function single_step!(ctx, step, sched, dataset, world)
    batches = Any[]
    for r in 0:(world - 1)
        head = sched.heads[step, r + 1]
        idx = sched.samples[step, r + 1]
        pool = dataset.pools[head]
        push!(batches, (head, to_target(pool.train.x[:, idx]),
            to_target(pool.train.y[:, idx])))
    end
    present = CGSchedule.schedule_present(sched, step)
    result = CGReference.reference_step!(ctx.model, ctx.tree, batches, present)
    return merge(ctx, (; model = result.model, tree = result.tree))
end

# single: timed step. Builds the same ordered batch list as `single_step!` but
# calls the plain native `reference_update!` with no host snapshots and no
# assertions, so the single baseline measures the same update work that DDP
# measures instead of diagnostic Array copies.
function single_timed_step!(ctx, step, sched, dataset, world)
    batches = Any[]
    for r in 0:(world - 1)
        head = sched.heads[step, r + 1]
        idx = sched.samples[step, r + 1]
        pool = dataset.pools[head]
        push!(batches, (head, to_target(pool.train.x[:, idx]),
            to_target(pool.train.y[:, idx])))
    end
    result = CGReference.reference_update!(ctx.model, ctx.tree, batches)
    return merge(ctx, (; model = result.model, tree = result.tree))
end

# ddp: one local shard per rank through the real wrapper backend; the wrapper
# allreduce stays outside AD. The same function serves timed and untimed steps
# because it never takes host snapshots.
function ddp_step!(ctx, step, sched, dataset, rank)
    head = sched.heads[step, rank + 1]
    idx = sched.samples[step, rank + 1]
    pool = dataset.pools[head]
    x = to_target(pool.train.x[:, idx])
    y = to_target(pool.train.y[:, idx])
    _, gs_box = Zygote.withgradient(m -> CGReference.local_loss(m, x, y, head), ctx.model)
    gs = gs_box[1]
    state, model = Optimisers.update!(ctx.state, ctx.model, gs)
    return merge(ctx, (; model, state, tree = state.tree))
end

build_single(prepared) = (; model = to_target_model(prepared.model),
    tree = to_target_tree(prepared.tree), state = nothing)

function build_ddp(prepared, backend)
    tree = to_target_tree(prepared.tree)
    return (; model = to_target_model(prepared.model), tree, state = W(backend, tree))
end

# ---------------------------------------------------------------------------
# Trajectory check (never part of a timed measurement)
# ---------------------------------------------------------------------------

function check_trajectory!(; mode, build, step!, trajectory, check_steps, world,
        rank, comm, tol)
    trajectory.steps >= check_steps ||
        error("saved trajectory covers $(trajectory.steps) steps, need $check_steps")
    ctx = build()
    local_ok = true
    local_max = 0.0
    for step in 1:check_steps
        ctx = step!(ctx, step)
        snap = CGCommon.snapshot(ctx.model, ctx.tree)
        expected = get(trajectory.snapshots, step, nothing)
        expected === nothing && error("saved trajectory has no snapshot for step $step")
        length(snap.values) == length(expected) ||
            error("step $step: snapshot length $(length(snap.values)) != trajectory $(length(expected))")
        maxabs = 0.0
        nbad = 0
        @inbounds for i in eachindex(snap.values)
            a = snap.values[i]
            b = expected[i]
            e = abs(a - b)
            maxabs = max(maxabs, e)
            nbad += e <= tol.atol + tol.rtol * abs(b) ? 0 : 1
        end
        local_max = max(local_max, maxabs)
        local_ok &= nbad == 0
        println(@sprintf("TIMING_CHECK step=%d maxabs=%.6e nbad=%d", step, maxabs, nbad))
    end
    if mode == "ddp"
        global_max = MPI.Allreduce(local_max, MPI.MAX, comm)
        ok = MPI.Allreduce(Int(local_ok), MPI.LAND, comm) == 1
    else
        global_max = local_max
        ok = local_ok
    end
    rank == 0 &&
        println(@sprintf("TIMING_CHECK_RESULT source=trajectory %s maxabs=%.6e",
            ok ? "PASS" : "FAIL", global_max))
    return ok
end

# ---------------------------------------------------------------------------
# Campaign: warm-up, checks, timed repeats
# ---------------------------------------------------------------------------

function run_campaign(; mode, label, build, step!, timed_step!, schedule, dataset, world, rank,
        comm, on_gpu, out_abs, run_id, steps, warmup, repeats, record_every,
        tol, trajectory, check_steps, dump_final, write_outputs)
    records_path = joinpath(out_abs, "records.tsv")
    steps_path = joinpath(out_abs, "steps.tsv")
    repeats_path = joinpath(out_abs, "repeats.tsv")
    if write_outputs
        write_header(records_path, RECORD_COLUMNS)
        write_header(steps_path, STEP_COLUMNS)
        write_header(repeats_path, REPEAT_COLUMNS)
    end
    ddp = mode == "ddp"
    barrier() = ddp && MPI.Barrier(comm)

    # Initial replica identity. Every rank must start from the same loaded
    # state; this is outside every timed region.
    if ddp
        ctx = build()
        initial = CGCommon.snapshot(ctx.model, ctx.tree).values
        all_initial = MPI.Allgather(initial, comm)
        for r in 0:(world - 1)
            CGCommon.ensure(all_initial[r * length(initial) + 1:(r + 1) * length(initial)] == initial,
                "initial state differs across ranks")
        end
    end

    # ----- untimed warm-up; state and schedule position are discarded -----
    # The warm-up mirrors the timed path (`timed_step!`), so no host-snapshot
    # diagnostic code is compiled into the measured region either.
    warm0 = time_ns()
    ctx = build()
    for step in 1:warmup
        ctx = timed_step!(ctx, step)
    end
    if rank == 0
        validate_model(ctx.model, dataset)
    end
    barrier()
    compile_warmup_s = (time_ns() - warm0) / 1e9
    rank == 0 && println(@sprintf("TIMING_WARMUP mode=%s warmup_steps=%d compile_warmup_s=%.6f",
        mode, warmup, compile_warmup_s))

    # ----- correctness check (separate, never timed) -----
    if trajectory !== nothing
        ok = check_trajectory!(; mode, build, step!, trajectory, check_steps,
            world, rank, comm, tol)
        barrier()
        ok || error("trajectory check failed")
    end

    # ----- timed repeats -----
    # `t_total_s` is the rank-local wall time of the repeat. Training-only
    # seconds accumulate around `timed_step!` alone; validation and record
    # writes run inside a barrier-guarded diagnostic window that every rank
    # enters and leaves in the same order, so they never leak into
    # `t_train_s`. The slowest-rank values come from separate Allgathers of
    # the local elapsed and local training nanoseconds.
    repeats_total = Float64[]
    repeats_train = Float64[]
    repeats_slowest = Float64[]
    repeats_slowest_train = Float64[]
    final_ctx = nothing
    for repeat in 1:repeats
        ctx = build()
        barrier()
        on_gpu && CUDA.synchronize()
        t0 = time_ns()
        local_train_ns = Int64(0)
        t_val_ns = Int64(0)
        t_write_ns = Int64(0)
        step_rows = NamedTuple[]
        val_latest = nothing
        samples_a = 0
        samples_b = 0
        for step in 1:steps
            on_gpu && CUDA.synchronize()
            seg0 = time_ns()
            ctx = timed_step!(ctx, step)
            on_gpu && CUDA.synchronize()
            local_train_ns += time_ns() - seg0
            now = time_ns()
            samples_a += schedule.task_requests[step, 1]
            samples_b += schedule.task_requests[step, 2]
            if rank == 0
                push!(step_rows, (; step,
                    elapsed_s = (now - t0) / 1e9,
                    train_s = local_train_ns / 1e9,
                    samples_total = step * schedule.global_batch,
                    samples_a, samples_b))
            end
            if step % record_every == 0
                barrier()
                if rank == 0
                    v0 = time_ns()
                    val_latest = validate_model(ctx.model, dataset)
                    t_val_ns += time_ns() - v0
                    w0 = time_ns()
                    append_flush(records_path, Any[run_id, mode, repeat, step,
                        @sprintf("%.6f", (now - t0) / 1e9),
                        @sprintf("%.6f", local_train_ns / 1e9),
                        step * schedule.global_batch, samples_a, samples_b,
                        @sprintf("%.6f", step / (local_train_ns / 1e9)),
                        @sprintf("%.6f", (step * schedule.global_batch) / (local_train_ns / 1e9)),
                        @sprintf("%.9e", val_latest.a.loss), @sprintf("%.9f", val_latest.a.accuracy),
                        @sprintf("%.9e", val_latest.b.loss), @sprintf("%.9f", val_latest.b.accuracy)])
                    t_write_ns += time_ns() - w0
                end
                barrier()
            end
        end
        on_gpu && CUDA.synchronize()
        local_elapsed_ns = time_ns() - t0
        barrier()
        slowest_s = ddp ? maximum(MPI.Allgather(local_elapsed_ns, comm)) / 1e9 :
                    local_elapsed_ns / 1e9
        slowest_train_s = ddp ? maximum(MPI.Allgather(local_train_ns, comm)) / 1e9 :
                           local_train_ns / 1e9
        t_total_s = local_elapsed_ns / 1e9
        t_train_s = local_train_ns / 1e9
        t_val_s = t_val_ns / 1e9
        t_write_s = t_write_ns / 1e9
        samples_total = steps * schedule.global_batch
        if rank == 0
            for row in step_rows
                append_flush(steps_path, Any[mode, repeat, row.step,
                    @sprintf("%.6f", row.elapsed_s), @sprintf("%.6f", row.train_s),
                    row.samples_total, row.samples_a, row.samples_b])
            end
            append_flush(repeats_path, Any[run_id, mode, repeat,
                @sprintf("%.6f", t_total_s), @sprintf("%.6f", t_train_s),
                @sprintf("%.6f", t_val_s), @sprintf("%.6f", t_write_s),
                @sprintf("%.6f", slowest_s), @sprintf("%.6f", slowest_train_s),
                steps, samples_total,
                @sprintf("%.6f", steps / slowest_s), @sprintf("%.6f", samples_total / slowest_s),
                @sprintf("%.6f", steps / slowest_train_s),
                @sprintf("%.6f", samples_total / slowest_train_s)])
            println(@sprintf("TIMING_REPEAT mode=%s repeat=%d t_total_s=%.6f t_train_s=%.6f t_val_s=%.6f t_write_s=%.6f slowest_rank_s=%.6f slowest_train_s=%.6f",
                mode, repeat, t_total_s, t_train_s, t_val_s, t_write_s, slowest_s, slowest_train_s))
        end
        push!(repeats_total, t_total_s)
        push!(repeats_train, t_train_s)
        push!(repeats_slowest, slowest_s)
        push!(repeats_slowest_train, slowest_train_s)
        final_ctx = ctx
    end

    if rank == 0
        total_series = ddp ? repeats_slowest : repeats_total
        train_series = ddp ? repeats_slowest_train : repeats_train
        median_total_s = median_of(total_series)
        median_train_s = median_of(train_series)
        println(@sprintf("TIMING_SUMMARY mode=%s repeats=%d median_total_s=%.6f min_total_s=%.6f max_total_s=%.6f median_train_s=%.6f min_train_s=%.6f max_train_s=%.6f median_updates_per_s=%.6f median_train_updates_per_s=%.6f",
            mode, repeats, median_total_s, minimum(total_series), maximum(total_series),
            median_train_s, minimum(train_series), maximum(train_series),
            steps / median_total_s, steps / median_train_s))
    end

    # Final state equality and optional dump. Rank 0 writes every file,
    # including the per-rank diagnostic dumps it builds from an Allgather, so
    # the bounded driver can compare replicas and other ranks never write.
    if dump_final !== nothing
        snap = CGCommon.snapshot(final_ctx.model, final_ctx.tree)
        if ddp
            flat = MPI.Allgather(snap.values, comm)
            n = length(snap.values)
            for r in 0:(world - 1)
                CGCommon.ensure(flat[r * n + 1:(r + 1) * n] == snap.values,
                    "final state differs across ranks")
            end
            if rank == 0
                for r in 0:(world - 1)
                    payload_r = (; mode, world, steps, done_at = string(Dates.now()),
                        values = flat[r * n + 1:(r + 1) * n], layout = snap.layout)
                    open("$dump_final.rank$(r).jls", "w") do io
                        serialize(io, payload_r)
                    end
                end
            end
        end
        if rank == 0
            payload = (; mode, world, steps, done_at = string(Dates.now()),
                values = snap.values, layout = snap.layout)
            open(dump_final, "w") do io
                serialize(io, payload)
            end
        end
    end

    println(@sprintf("TIMING_RANK_DONE mode=%s rank=%d steps=%d", mode, rank, steps))
    barrier()
    if rank == 0
        open(joinpath(out_abs, "done"), "w") do io
            println(io, "$run_id $mode status=PASS")
        end
        println("TIMING_DONE mode=$mode")
    end
    barrier()
    return compile_warmup_s
end

# ---------------------------------------------------------------------------
# Mode drivers
# ---------------------------------------------------------------------------

function load_prepared(prepared_path)
    isfile(prepared_path) || error("prepared artifact $prepared_path does not exist")
    prepared = open(deserialize, prepared_path)
    prepared isa NamedTuple || error("prepared artifact is not a NamedTuple")
    prepared.kind in (:phase_k_prepared, :phase_k_toy_prepared) ||
        error("$prepared_path is not a Phase K prepared artifact")
    CGCommon.ensure(CGDataset.dataset_digest(prepared.dataset.pools) == prepared.dataset_digest,
        "prepared dataset digest mismatch")
    return prepared
end

function load_trajectory(path)
    isfile(path) || error("trajectory $path does not exist")
    trajectory = open(deserialize, path)
    return trajectory
end

function check_run_identity(prepared, trajectory, schedule, tol)
    if trajectory !== nothing
        if hasproperty(trajectory, :schedule_digest)
            CGCommon.ensure(trajectory.schedule_digest == CGSchedule.schedule_digest(schedule),
                "trajectory schedule digest $(trajectory.schedule_digest) does not match " *
                "schedule digest $(CGSchedule.schedule_digest(schedule))")
        end
        if hasproperty(trajectory, :dataset_digest)
            CGCommon.ensure(trajectory.dataset_digest == prepared.dataset_digest,
                "trajectory dataset digest does not match the prepared artifact")
        end
        tol_in_traj = hasproperty(trajectory, :tolerances) ? trajectory.tolerances : nothing
        if tol_in_traj isa CGCommon.Tolerances && tol_in_traj != tol
            @warn "declared tolerances differ from the saved trajectory" tol tol_in_traj
        end
    end
end

function main_single(opts, prepared, schedule, world, trajectory)
    out = require_opt(opts, "out")
    label = get(opts, "label", "phase-k-timing")
    steps = int_opt(opts, "steps", 120)
    warmup = int_opt(opts, "warmup", 3)
    repeats = int_opt(opts, "repeats", 3)
    record_every = int_opt(opts, "record-every", 5)
    check_steps = int_opt(opts, "check-steps", 12)
    dump_final = get(opts, "dump-final", nothing)
    device = get(opts, "device", "cpu")
    device in ("cpu", "cuda") || error("--device must be cpu or cuda")
    ON_GPU[] = device == "cuda"
    TARGET_DTYPE[] = ON_GPU[] ? Float32 : eltype(prepared.model.trunk.weight)
    if ON_GPU[]
        CUDA.allowscalar(false)
        CUDA.functional() || error("--device cuda requested but CUDA is not functional")
    end
    validate_workload(steps, warmup, repeats, record_every, check_steps, trajectory)
    tol = resolve_tolerances(opts, TARGET_DTYPE[])
    check_run_identity(prepared, trajectory, schedule, tol)

    out_abs = abspath(out)
    isfile(joinpath(out_abs, "records.tsv")) && error("refusing existing run output $out_abs")
    run_id = get(opts, "run-id", basename(out_abs))
    mkpath(out_abs)

    eta = prepared.tree.trunk.weight.rule.eta
    device_rows = [(0, ON_GPU[] ? CUDA.deviceid(CUDA.device()) : -1)]
    extra = (; on_gpu = ON_GPU[],
        device_name = ON_GPU[] ? CUDA.name(CUDA.device()) : "cpu",
        check_trajectory = trajectory === nothing ? "none" : abspath(get(opts, "check-trajectory", "")))
    text = metadata_text(; mode = "single", label, backend = device, world, rank = 0,
        dtype = string(TARGET_DTYPE[]), tol, sched = schedule, dataset = prepared.dataset,
        prepared_path = require_opt(opts, "prepared"),
        prepared_sha = CGCommon.source_hash(require_opt(opts, "prepared")),
        run_id, steps, warmup, repeats, record_every, eta, device_rows, extra)
    CGCommon.write_text(joinpath(out_abs, "metadata.txt"), text)
    println("TIMING_META mode=single rank=0 world=$world dtype=$(TARGET_DTYPE[]) " *
            "tolerances=\"$tol\" flux=$(pathof(Flux)) schedule_digest=$(CGSchedule.schedule_digest(schedule))")
    println("TIMING_START mode=single world=$world steps=$steps warmup=$warmup " *
            "repeats=$repeats record_every=$record_every device=$device")

    compile_warmup_s = run_campaign(; mode = "single", label,
        build = () -> build_single(prepared),
        step! = (ctx, step) -> single_step!(ctx, step, schedule, prepared.dataset, world),
        timed_step! = (ctx, step) -> single_timed_step!(ctx, step, schedule, prepared.dataset, world),
        schedule, dataset = prepared.dataset, world, rank = 0, comm = nothing,
        on_gpu = ON_GPU[], out_abs, run_id, steps, warmup, repeats, record_every, tol,
        trajectory, check_steps, dump_final, write_outputs = true)
    append_flush(joinpath(out_abs, "metadata.txt"), Any["extra", "compile_warmup_s",
        @sprintf("%.6f", compile_warmup_s)])
    return nothing
end

function main_ddp(opts, prepared, schedule, world, trajectory)
    out = require_opt(opts, "out")
    label = get(opts, "label", "phase-k-timing")
    steps = int_opt(opts, "steps", 120)
    warmup = int_opt(opts, "warmup", 3)
    repeats = int_opt(opts, "repeats", 3)
    record_every = int_opt(opts, "record-every", 5)
    check_steps = int_opt(opts, "check-steps", 12)
    dump_final = get(opts, "dump-final", nothing)
    backend_name = get(opts, "backend", "mpi")
    backend_name in ("mpi", "nccl") || error("--backend must be mpi or nccl")
    ON_GPU[] = backend_name == "nccl"
    TARGET_DTYPE[] = ON_GPU[] ? Float32 : eltype(prepared.model.trunk.weight)
    validate_workload(steps, warmup, repeats, record_every, check_steps, trajectory)
    tol = resolve_tolerances(opts, TARGET_DTYPE[])

    ctx = init_backend(backend_name)
    (; backend, rank, workers, comm) = ctx
    workers >= 2 || error("the conditional schedule needs at least 2 ranks")
    workers == world || error("backend has $workers ranks, --world says $world")
    schedule = get(prepared.schedules, workers, nothing)
    schedule === nothing && error("prepared artifact has no schedule for world size " *
                                  "$workers (available: $(sort(collect(keys(prepared.schedules)))))")
    CGSchedule.validate_schedule(schedule)
    check_run_identity(prepared, trajectory, schedule, tol)

    out_abs = abspath(out)
    refuse_flag = MPI.Bcast((rank == 0 && isfile(joinpath(out_abs, "records.tsv"))) ? 1 : 0,
        0, comm)
    refuse_flag == 1 && error("refusing existing run output $out_abs")
    run_id = get(opts, "run-id", basename(out_abs))
    rank == 0 && mkpath(out_abs)
    MPI.Barrier(comm)

    eta = prepared.tree.trunk.weight.rule.eta
    device = ON_GPU[] ? CUDA.deviceid(CUDA.device()) : -1
    all_devices = MPI.Allgather([device], comm)
    if ON_GPU[]
        CGCommon.ensure(length(unique(all_devices)) == workers,
            "NCCL ranks are not on distinct physical devices: $(collect(all_devices))")
    end
    device_rows = [(r, all_devices[r + 1]) for r in 0:(workers - 1)]
    extra = (; on_gpu = ON_GPU[],
        device_name = ON_GPU[] ? CUDA.name(CUDA.device()) : "cpu",
        check_trajectory = trajectory === nothing ? "none" : abspath(get(opts, "check-trajectory", "")))
    if rank == 0
        text = metadata_text(; mode = "ddp", label, backend = backend_name, world,
            rank = 0, dtype = string(TARGET_DTYPE[]), tol, sched = schedule,
            dataset = prepared.dataset, prepared_path = require_opt(opts, "prepared"),
            prepared_sha = CGCommon.source_hash(require_opt(opts, "prepared")), run_id,
            steps, warmup, repeats, record_every, eta, device_rows, extra)
        CGCommon.write_text(joinpath(out_abs, "metadata.txt"), text)
        println("TIMING_META mode=ddp rank=0 backend=$backend_name ranks=$workers " *
                "dtype=$(TARGET_DTYPE[]) tolerances=\"$tol\" flux=$(pathof(Flux)) " *
                "schedule_digest=$(CGSchedule.schedule_digest(schedule))")
        println("TIMING_START mode=ddp world=$workers steps=$steps warmup=$warmup " *
                "repeats=$repeats record_every=$record_every backend=$backend_name")
    end
    compile_warmup_s = run_campaign(; mode = "ddp", label,
        build = () -> build_ddp(prepared, backend),
        step! = (ctx, step) -> ddp_step!(ctx, step, schedule, prepared.dataset, rank),
        timed_step! = (ctx, step) -> ddp_step!(ctx, step, schedule, prepared.dataset, rank),
        schedule, dataset = prepared.dataset, world = workers, rank, comm,
        on_gpu = ON_GPU[], out_abs, run_id, steps, warmup, repeats, record_every, tol,
        trajectory, check_steps, dump_final, write_outputs = rank == 0)
    if rank == 0
        append_flush(joinpath(out_abs, "metadata.txt"), Any["extra", "compile_warmup_s",
            @sprintf("%.6f", compile_warmup_s)])
    end
    return nothing
end

function validate_workload(steps, warmup, repeats, record_every, check_steps, trajectory)
    steps >= 1 || error("--steps must be >= 1")
    warmup >= 0 || error("--warmup must be >= 0")
    repeats >= 1 || error("--repeats must be >= 1")
    1 <= record_every <= steps || error("--record-every must be in 1:steps")
    check_steps >= 1 || error("--check-steps must be >= 1")
    return nothing
end

function resolve_tolerances(opts, dtype)
    base = CGCommon.tolerances_for(dtype)
    return CGCommon.Tolerances(float_opt(opts, "atol", base.atol),
        float_opt(opts, "rtol", base.rtol))
end

# ---------------------------------------------------------------------------

function main()
    MODE in ("single", "ddp") ||
        error("unknown mode $(repr(MODE)); expected single or ddp")
    opts, flags = parse_options(_RAW_ARGS)
    if haskey(opts, "out")
        _RUN_OUT[] = abspath(opts["out"])
        _RUN_ID[] = get(opts, "run-id", basename(_RUN_OUT[]))
    end
    _RUN_MODE[] = MODE

    expected_flux = get(opts, "flux-root",
        normpath(joinpath(@__DIR__, "../../../flux-integration")))
    realpath(dirname(dirname(pathof(Flux)))) == realpath(expected_flux) ||
        error("Flux resolves to $(pathof(Flux)), expected $expected_flux; " *
              "use the implementation test environment")

    prepared_path = require_opt(opts, "prepared")
    prepared = load_prepared(prepared_path)
    world = int_opt(opts, "world", 4)
    record_every = int_opt(opts, "record-every", 5)
    steps = int_opt(opts, "steps", 120)
    trajectory_path = get(opts, "check-trajectory", nothing)
    trajectory = trajectory_path === nothing ? nothing : load_trajectory(trajectory_path)

    schedule = get(prepared.schedules, world, nothing)
    schedule === nothing && error("prepared artifact has no schedule for world size " *
                                  "$world (available: $(sort(collect(keys(prepared.schedules)))))")
    CGSchedule.validate_schedule(schedule)
    check_steps = int_opt(opts, "check-steps", 12)
    trajectory !== nothing && trajectory.steps < check_steps &&
        error("saved trajectory covers $(trajectory.steps) steps, --check-steps $check_steps")

    if MODE == "single"
        main_single(opts, prepared, schedule, world, trajectory)
    else
        main_ddp(opts, prepared, schedule, world, trajectory)
    end
    return nothing
end

const _RUN_OUT = Ref{Union{Nothing,String}}(nothing)
const _RUN_ID = Ref("timing")
const _RUN_MODE = Ref(MODE)

function write_failure(err)
    out = _RUN_OUT[]
    out === nothing && return
    # Rank 0 owns the failure marker; other ranks only exit nonzero.
    if MPI.Initialized()
        MPI.Comm_rank(MPI.COMM_WORLD) == 0 || return
    end
    try
        mkpath(out)
        open(joinpath(out, "done"), "w") do io
            println(io, "$(_RUN_ID[]) $(_RUN_MODE[]) status=FAIL $(sprint(showerror, err))")
        end
    catch
    end
    return
end

ok = try
    main()
    true
catch err
    showerror(stderr, err)
    println(stderr)
    write_failure(err)
    false
end

exit(ok ? 0 : 1)

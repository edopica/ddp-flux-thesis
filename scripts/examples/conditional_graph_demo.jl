#!/usr/bin/env julia
# Phase K conditional-graph distributed demo CLI (toy dataset).
#
# Modes:
#   prepare  single process; writes the dataset, saved schedule, initial model,
#            and native Adam tree artifact (no MPI backend).
#   train    MPI/NCCL job; compares every distributed update against the
#            independent native reference on rank 0 and the saved schedule.
#   resume   fresh MPI/NCCL processes; loads a data-only checkpoint, restores
#            the native tree with `DistributedOptimizerState`, and continues
#            against the saved uninterrupted trajectory and the reference.
#
# Run under the Flux implementation environment, for example:
#   julia --project=../flux-integration/test scripts/examples/conditional_graph_demo.jl ...
#
# The demo uses only public Flux/Functors/Zygote/Optimisers APIs. Every
# collective issued by the wrapper is outside AD and recorded by a wrapper
# backend; diagnostics use the real backend and are recorded separately.

import Serialization: serialize, deserialize

const _RAW_ARGS = copy(ARGS)

function _raw_option(args, key, default=nothing)
    i = findfirst(==("--" * key), args)
    i === nothing && return default
    i < length(args) || error("option --$key requires a value")
    return args[i + 1]
end

const MODE = isempty(_RAW_ARGS) ? error("usage: conditional_graph_demo.jl " *
                                        "prepare|train|resume [options]") : _RAW_ARGS[1]
const BACKEND_ARG = _raw_option(_RAW_ARGS, "backend", "mpi")

# GPU mode must fail clearly when unavailable: CUDA/NCCL/cuDNN are loaded only
# for the explicit NCCL backend, and initialization later forces CUDA.
if BACKEND_ARG == "nccl"
    import CUDA
    import NCCL
    import cuDNN
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
include(joinpath(@__DIR__, "conditional_graph", "checkpoint.jl"))
include(joinpath(@__DIR__, "conditional_graph", "metrics.jl"))

using .CGCommon, .CGModel, .CGDataset, .CGSchedule, .CGReference, .CGCheckpoint

const ROUTING_CYCLE = [
    (; cycle = 1, rank0 = :a, others = :b, purpose = "local absence and one-versus-rest weighting"),
    (; cycle = 2, rank0 = :a, others = :b, purpose = "nonzero Adam history on both heads"),
    (; cycle = 3, all = :a, purpose = "B globally absent after prior activity"),
    (; cycle = 4, all = :b, purpose = "A globally absent after prior activity"),
    (; cycle = 5, rank0 = :a, others = :b, purpose = "reactivation after absence"),
    (; cycle = 6, rank0 = :b, others = :a, purpose = "reversed participation"),
]

const METRIC_COLUMNS = CGMetrics.DISCREPANCY_COLUMNS

const BOOLEAN_OPTIONS = Set(["save-trajectory", "help"])

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

function resolve_tolerances(opts, dtype)
    base = CGCommon.tolerances_for(dtype)
    return CGCommon.Tolerances(float_opt(opts, "atol", base.atol),
        float_opt(opts, "rtol", base.rtol))
end

is_on_gpu() = BACKEND_ARG == "nccl"
const TARGET_DTYPE = Ref{DataType}(Float64)

# ---------------------------------------------------------------------------
# Device-generic conversion helpers
# ---------------------------------------------------------------------------

function to_target(x::AbstractArray{<:Real})
    return is_on_gpu() ? CUDA.cu(Float32.(x)) : TARGET_DTYPE[].(x)
end

convert_array(x::AbstractArray) = to_target(x)
convert_array(x) = x

to_target_model(m) = Functors.fmap(convert_array, m)
to_target_tree(t) = Functors.fmap(convert_array, t)

host_copy(x::AbstractArray) = x isa Array ? copy(x) : Array(x)
host_copy(x) = x

hostify_model(m) = Functors.fmap(host_copy, m)
hostify_tree(t) = Functors.fmap(host_copy, t)

copy_model(m) = Functors.fmap(x -> x isa AbstractArray ? copy(x) : x, m)

# ---------------------------------------------------------------------------
# Backend initialization (explicit; no automatic CPU fallback)
# ---------------------------------------------------------------------------

function init_backend(backend_name::AbstractString)
    if backend_name == "nccl"
        DU.initialize(DU.NCCLBackend)
        backend = DU.get_distributed_backend(DU.NCCLBackend)
        on_gpu = true
        CUDA.allowscalar(false)
        CUDA.functional() || error("NCCL backend requested but CUDA is not functional")
    elseif backend_name == "mpi"
        DU.initialize(DU.MPIBackend; cuda_devices = missing, amdgpu_devices = missing)
        backend = DU.get_distributed_backend(DU.MPIBackend)
        on_gpu = false
    else
        error("unknown backend $(repr(backend_name)); expected \"mpi\" or \"nccl\"")
    end
    rank = DU.local_rank(backend)
    workers = DU.total_workers(backend)
    rec = CGCommon.RecordingBackend(backend, Any[])
    comm = CGCommon.diagnostic_comm(backend)
    if on_gpu
        device = CUDA.deviceid(CUDA.device())
        devices = MPI.Allgather([device], comm)
        CGCommon.ensure(length(unique(devices)) == workers,
            "NCCL ranks are not on distinct physical devices: $(collect(devices))")
        println("DEMO_GPU_MAP rank=$rank device=$device name=$(CUDA.name(CUDA.device())) " *
                "devices=$(collect(devices))")
    end
    return (; backend, rec, rank, workers, comm, on_gpu)
end

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

function demo_source_hashes()
    demo_dir = joinpath(@__DIR__, "conditional_graph")
    files = Dict{String,String}()
    for f in ("common.jl", "model.jl", "dataset.jl", "schedule.jl",
        "reference.jl", "checkpoint.jl", "metrics.jl")
        files["scripts/examples/conditional_graph/$f"] =
            CGCommon.source_hash(joinpath(demo_dir, f))
    end
    files["scripts/examples/conditional_graph_demo.jl"] = CGCommon.source_hash(@__FILE__)
    checker = normpath(joinpath(@__DIR__, "..", "checks", "conditional_graph_demo_check.jl"))
    isfile(checker) &&
        (files["scripts/checks/conditional_graph_demo_check.jl"] = CGCommon.source_hash(checker))
    flux_root = normpath(joinpath(dirname(pathof(Flux)), ".."))
    for rel in ("src/Flux.jl", "src/distributed/public_api.jl")
        p = joinpath(flux_root, rel)
        isfile(p) && (files["flux-integration/$rel"] = CGCommon.source_hash(p))
    end
    return files
end

function metadata_text(; mode, label, backend, workers, dtype, tol, sched, dataset,
        extra::NamedTuple)
    io = IOBuffer()
    println(io, "mode\t", mode)
    println(io, "label\t", label)
    println(io, "date\t", Dates.now())
    println(io, "host\t", CGCommon.host_name())
    println(io, "julia\t", VERSION)
    println(io, "backend\t", backend)
    println(io, "workers\t", workers)
    println(io, "dtype\t", dtype)
    println(io, "command\t", Base.julia_cmd(), " ", join(_RAW_ARGS, ' '))
    println(io, "diagnostics\tMPI Allgather configuration; Bcast reference configuration; Allgather snapshots; Bcast reference snapshot; Allgather step; checkpoint barrier")
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
    println(io, "tolerances\t", tol)
    for pkg in CGCommon.package_versions()
        println(io, "package\t$(pkg.name)\t$(pkg.version)\t$(pkg.path)")
    end
    println(io, "package\tZygote\t$(pkgversion(Zygote))\t$(pathof(Zygote))")
    if is_on_gpu()
        for pkg in (CUDA, NCCL, cuDNN)
            println(io, "package\t$(nameof(pkg))\t$(pkgversion(pkg))\t$(pathof(pkg))")
        end
    end
    println(io, "dataset\t$(dataset.name)\tdigest=$(dataset.digest)" *
                "\ttrain_per_task=$(dataset.train_per_task)\tval_per_task=$(dataset.val_per_task)")
    if sched !== nothing
        println(io, "schedule\tdigest=$(CGSchedule.schedule_digest(sched))" *
                    "\tsteps=$(sched.steps)\tglobal_batch=$(sched.global_batch)" *
                    "\tlocal_batch=$(sched.local_batch)\tworkers=$(sched.workers)")
        for t in CGSchedule.schedule_summary(sched).tasks
            println(io, "schedule_task\t$(t.task)\texposure=$(t.exposure)\tpool=$(t.pool)" *
                        "\tcompleted_passes=$(t.completed_passes)\tpartial=$(t.partial)" *
                        "\treused=$(t.reused)")
        end
    end
    for (k, v) in sort(collect(pairs(extra)); by = first)
        println(io, "extra\t$k\t$v")
    end
    for (f, h) in sort(collect(demo_source_hashes()); by = first)  # Dict iteration yields Pairs
        println(io, "source_hash\t$f\t$h")
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Reference-comparison run loop
# ---------------------------------------------------------------------------

function validate_task(model, dataset, task::Symbol, on_gpu::Bool)
    pool = dataset.pools[task].val
    n = size(pool.y, 2)
    total_loss = 0.0
    correct = 0
    for first in 1:128:n
        ids = first:min(first+127, n)
        x = to_target(pool.x[:, ids]); y = to_target(pool.y[:, ids])
        logits = model(x, task)
        loss = Flux.Losses.logitcrossentropy(logits, y)
        loss isa Number || error("validation loss is not a host scalar")
        total_loss += Float64(loss) * length(ids)
        # Explicit diagnostic transfer, outside AD and communication.
        correct += sum(argmax(Array(logits); dims=1) .== argmax(pool.y[:, ids]; dims=1))
    end
    return (; loss=total_loss/n, accuracy=correct/n, n)
end

function write_validation(out, mode, run_id, sched, dataset, step, model, ref_model;
        checkpoint_step=2, predictions=false)
    digest = CGSchedule.schedule_digest(sched)
    for (name, m) in (("ddp", model), ("reference", ref_model)), (ti, task) in enumerate((:a, :b))
        v = validate_task(m, dataset, task, is_on_gpu())
        exposure = sum(sched.task_requests[1:step, ti])
        present = step == 0 || sched.task_requests[step, ti] > 0
        CGMetrics.append_row(out, "validation", mode, (run_id, mode, digest,
            dataset.digest, step, name, task, v.n, v.loss, v.accuracy,
            exposure, present, checkpoint_step))
        if predictions
            pool = dataset.pools[task].val
            ids = 1:min(20, size(pool.x, 2))
            logits = Array(m(to_target(pool.x[:, ids]), task))
            for i in ids
                official = hasproperty(pool, :indices) ? pool.indices[i] : i
                CGMetrics.append_row(out, "predictions", mode, (run_id, mode, digest,
                    step, name, task, official, argmax(pool.y[:, i])-1,
                    argmax(logits[:, i])-1, join(Float64.(pool.x[:, i]), ',')))
            end
        end
    end
end

function check_structure(model, tree, grads=nothing)
    for (sub, field) in CGCommon.LEAF_ORDER
        p = getproperty(getproperty(model, sub), field)
        leaf = getproperty(getproperty(tree, sub), field)
        for a in (p, leaf.state[1], leaf.state[2])
            size(a) == size(p) || error("shape mismatch at $sub.$field")
            eltype(a) === TARGET_DTYPE[] || error("dtype mismatch at $sub.$field")
            (is_on_gpu() ? a isa CUDA.CuArray : a isa Array) || error("device mismatch at $sub.$field")
        end
        if grads !== nothing
            gsub = getproperty(grads, sub)
            if gsub !== nothing
                g = getproperty(gsub, field)
                size(g) == size(p) && eltype(g) === eltype(p) || error("gradient shape/dtype mismatch")
                (is_on_gpu() ? g isa CUDA.CuArray : g isa Array) || error("gradient device mismatch")
            end
        end
    end
end

function record_config(out, mode, run_id, digest, step, tree, ref_tree, rank, R, comm)
    config(t) = reduce(vcat, [begin
        leaf = getproperty(getproperty(t, sub), field)
        Float64[leaf.rule.eta, leaf.rule.beta[1], leaf.rule.beta[2], leaf.rule.epsilon, leaf.frozen]
    end for (sub, field) in CGCommon.LEAF_ORDER])
    local_config = config(tree)
    all_config = MPI.Allgather(local_config, comm)
    reference_config = rank == 0 ? config(ref_tree) : zeros(length(local_config))
    MPI.Bcast!(reference_config, comm; root=0)
    for r in 0:R-1
        all_config[r*30+1:(r+1)*30] == reference_config || error("step $step rank $r rule/frozen mismatch")
    end
    eta = reference_config[1]
    all(==(eta), reference_config[1:5:end]) || error("per-leaf eta mismatch")
    if rank == 0
        for (name, ranks) in (("ddp", 0:R-1), ("reference", 0:0)), r in ranks
            for (i, leaf) in enumerate(CGMetrics.LEAVES)
                value = name == "ddp" ? all_config[r*30+(i-1)*5+1] : reference_config[(i-1)*5+1]
                CGMetrics.append_row(out, "learning-rate", mode,
                    (run_id, mode, digest, step, name, r, leaf, value))
            end
        end
    end
    return eta
end

function run_steps(; mode, label, rank, R, on_gpu, backend, rec, comm, tol,
        sched, dataset, model, state, ref_model, ref_tree, out,
        start_step, stop_step, checkpoint_at, checkpoint_path, make_payload,
        save_trajectory, expected_values, trajectory_meta, run_id,
        expected_reference=nothing, checkpoint_step=2)
    metrics_path = joinpath(out, "metrics-$mode.tsv")
    if rank == 0
        CGMetrics.init_files(out, mode)
        open(metrics_path, "w") do io
            println(io, join(METRIC_COLUMNS, '\t'))
        end
    end

    snapshots = Dict{Int,Vector{Float64}}()
    ref_snapshots = Dict{Int,Vector{Float64}}()
    losses = Dict{Int,Float64}()
    checkpoint_digest = nothing
    last_layout = nothing
    final_row = nothing
    eta = state.tree.trunk.weight.rule.eta
    config_initial = deepcopy(state.tree)
    check_structure(model, state.tree)
    # Every replica must also start at the exact same loaded state.
    initial = CGCommon.snapshot(model, state.tree).values
    all_initial = MPI.Allgather(initial, comm)
    for r in 0:R-1
        CGCommon.ensure(all_initial[r*length(initial)+1:(r+1)*length(initial)] == initial,
            "initial state differs across ranks")
    end
    if rank == 0
        check_structure(ref_model, ref_tree)
        write_validation(out, mode, run_id, sched, dataset, start_step-1, model, ref_model;
            checkpoint_step, predictions=true)
    end

    for step in start_step:stop_step
        recorded_eta = record_config(out, mode, run_id, trajectory_meta.schedule_digest,
            step, state.tree, ref_tree, rank, R, comm)
        CGCommon.ensure(recorded_eta == eta, "fixed Adam eta changed")
        CGCommon.check_leaf_config(config_initial, state.tree; context="step $step rank $rank")
        cycle = CGSchedule.cycle_step(step)
        head_here = sched.heads[step, rank + 1]
        idx = sched.samples[step, rank + 1]
        pool = dataset.pools[head_here]
        x = to_target(pool.train.x[:, idx])
        y = to_target(pool.train.y[:, idx])

        loss_local_raw, gs_box = Zygote.withgradient(
            m -> CGReference.local_loss(m, x, y, head_here), model)
        gs = gs_box[1]
        check_structure(model, state.tree, gs)
        caller_grads = Functors.fmap(host_copy, gs)
        loss_local_raw isa Number || error("local loss is not a host scalar")
        loss_local = Float64(loss_local_raw)

        # Local AD presence: the unused head must be absent from this rank's
        # real AD graph.
        for h in (:a, :b)
            g = getproperty(gs, CGModel.head_field(h))
            used = (h == head_here)
            CGCommon.ensure(used ? g !== nothing : g === nothing,
                "step $step rank $rank: local AD gradient for head $h " *
                (used ? "is absent but the head was used" : "is present but the head was unused"))
        end

        present = CGSchedule.schedule_present(sched, step)
        counts = (; a = count(==(:a), view(sched.heads, step, :)),
            b = count(==(:b), view(sched.heads, step, :)))

        # Before a head becomes globally absent, snapshot its complete history
        # and require nonzero Adam moments.
        absent_head = nothing
        absent_data = nothing
        for h in (:a, :b)
            getproperty(present, h) && continue
            absent_head = h
            absent_data = CGCommon.head_snapshot(model, state.tree, h)
            CGCommon.ensure(any(!iszero, absent_data.mw) && any(!iszero, absent_data.vw) &&
                            any(!iszero, absent_data.mb) && any(!iszero, absent_data.vb),
                "step $step: head $h becomes globally absent without nonzero Adam history")
        end

        # Independent native reference on rank 0, outside every distributed
        # call. It never reads wrapper output and never resets to the
        # distributed replicas.
        loss_ref = NaN
        ref_snap = nothing
        if rank == 0
            batches = Any[]
            for r in 0:(R - 1)
                h = sched.heads[step, r + 1]
                idxr = sched.samples[step, r + 1]
                pr = dataset.pools[h]
                push!(batches, (h, to_target(pr.train.x[:, idxr]),
                    to_target(pr.train.y[:, idxr])))
            end
            loss_ref_raw, grads_ref = CGReference.global_gradient(ref_model, batches)
            loss_ref_raw isa Number || error("reference loss is not a host scalar")
            loss_ref = Float64(loss_ref_raw)
            check_structure(ref_model, ref_tree, grads_ref)
            for h in (:a, :b)
                g = getproperty(grads_ref, CGModel.head_field(h))
                want = getproperty(present, h)
                CGCommon.ensure(want ? g !== nothing : g === nothing,
                    "step $step reference: globally unused head $h is " *
                    (want ? "absent" : "present") * " in the native AD result")
            end
            ref_absent = absent_head === nothing ? nothing : CGCommon.head_snapshot(ref_model, ref_tree, absent_head)
            ref_tree, ref_model = Optimisers.update!(ref_tree, ref_model, grads_ref)
            if absent_head !== nothing
                CGCommon.ensure(CGCommon.head_snapshot(ref_model, ref_tree, absent_head) == ref_absent,
                    "reference absent head changed at step $step")
            end
            ref_snap = CGCommon.snapshot(ref_model, ref_tree)
        end

        # Wrapper update through the recording backend.
        empty!(rec.log)
        n_before = length(rec.log)
        state, model = Optimisers.update!(state, model, gs)
        CGCommon.ensure(Functors.fmap(host_copy, gs) == caller_grads, "caller gradients mutated at step $step")
        check_structure(model, state.tree)
        CGCommon.check_leaf_config(config_initial, state.tree; context="post-update step $step rank $rank")
        if rank == 0
            CGCommon.check_leaf_config(config_initial, ref_tree; context="post-update reference step $step")
        end
        step_log = rec.log[(n_before + 1):end]
        CGCommon.check_collective_trace(step_log, model, step, rank, head_here, present,
            counts, R, on_gpu)
        # Each rank emits its own trace to the captured console, never a shared file.
        for (i, entry) in enumerate(step_log)
            println("DEMO_COLLECTIVE mode=$mode step=$step rank=$rank order=$i type=$(entry.buftype) shape=$(entry.size) avg=$(entry.avg) local=$(entry.local_flags) global=$(entry.global_flags)")
        end

        # Exact preservation through a globally absent step.
        if absent_head !== nothing
            after = CGCommon.head_snapshot(model, state.tree, absent_head)
            CGCommon.ensure(after == absent_data,
                "step $step: globally absent head $absent_head changed during the skip step")
        end

        # Diagnostics on the real backend (separate from the wrapper trace):
        # gather every replica's host snapshot and broadcast rank 0's reference
        # snapshot to all ranks.
        local_snap = CGCommon.snapshot(model, state.tree)
        last_layout = local_snap.layout
        n_values = length(local_snap.values)
        flat = MPI.Allgather(local_snap.values, comm)
        all_values = [view(flat, ((j - 1) * n_values + 1):(j * n_values)) for j in 1:R]
        ref_values = rank == 0 ? ref_snap.values : zeros(Float64, n_values)
        MPI.Bcast!(ref_values, comm; root = 0)

        peer_param = 0.0
        peer_mom1 = 0.0
        peer_mom2 = 0.0
        peer_beta = true
        for j in 1:R
            c = CGCommon.compare_snapshots(all_values[j], ref_values, local_snap.layout, tol;
                context="step $step replica $(j-1) versus reference")
            peer_param = max(peer_param, c.param.maxabs)
            peer_mom1 = max(peer_mom1, c.mom1.maxabs)
            peer_mom2 = max(peer_mom2, c.mom2.maxabs)
            peer_beta &= c.beta_exact
            CGCommon.ensure(c.param.worst <= 1.0 && c.mom1.worst <= 1.0 &&
                            c.mom2.worst <= 1.0,
                "step $step observer rank $rank: replica rank $(j - 1) diverges from the " *
                "native reference ($(CGCommon.format_comparison(c)))")
            CGCommon.ensure(c.beta_exact,
                "step $step observer rank $rank: replica rank $(j - 1) Adam decay " *
                "products differ from the reference")
            peer = CGCommon.compare_snapshots(all_values[j], all_values[1], local_snap.layout, tol)
            CGCommon.ensure(peer.param.worst <= 1 && peer.mom1.worst <= 1 && peer.mom2.worst <= 1 && peer.beta_exact,
                "step $step rank $(j-1) differs from rank zero")
        end
        CGCommon.ensure(all(==(step), MPI.Allgather([step], comm)), "unequal update counts")

        expected_cmp = nothing
        if expected_values !== nothing
            exp = get(expected_values, step, nothing)
            exp === nothing && error("saved trajectory has no snapshot for step $step")
            expected_cmp = CGCommon.compare_snapshots(local_snap.values, exp,
                local_snap.layout, tol)
            CGCommon.ensure(expected_cmp.param.worst <= 1.0 && expected_cmp.mom1.worst <= 1.0 &&
                            expected_cmp.mom2.worst <= 1.0 && expected_cmp.beta_exact,
                "step $step rank $rank: resumed trajectory diverges from the saved " *
                "uninterrupted trajectory ($(CGCommon.format_comparison(expected_cmp)))")
        end

        if expected_reference !== nothing
            c = CGCommon.compare_snapshots(ref_values, expected_reference[step], local_snap.layout, tol)
            CGCommon.ensure(c.param.worst <= 1 && c.mom1.worst <= 1 && c.mom2.worst <= 1 && c.beta_exact,
                "step $step resumed reference differs from uninterrupted reference")
        end

        println(@sprintf("DEMO_STEP mode=%s step=%d cycle=%d rank=%d head=%s present_a=%s present_b=%s counts_a=%d counts_b=%d loss_ref=%.10f loss_local=%.10f peer_param_maxerr=%.3e peer_mom1_maxerr=%.3e peer_mom2_maxerr=%.3e beta_exact=%s collectives=%d resume_match=%s",
            mode, step, cycle, rank, head_here, present.a, present.b, counts.a, counts.b,
            loss_ref, loss_local, peer_param, peer_mom1, peer_mom2, peer_beta,
            length(step_log), expected_cmp === nothing ? "n/a" : string(expected_cmp.param.worst <= 1.0)))

        if rank == 0
            write_validation(out, mode, run_id, sched, dataset, step, model, ref_model;
                checkpoint_step, predictions=(step == stop_step))
            val_a = validate_task(model, dataset, :a, on_gpu)
            val_b = validate_task(model, dataset, :b, on_gpu)
            row = (; step, cycle, present_a = present.a, present_b = present.b,
                counts_a = counts.a, counts_b = counts.b, loss_ref, loss_local,
                peer_param, peer_mom1, peer_mom2, peer_beta,
                val_a_loss = val_a.loss, val_a_acc = val_a.accuracy,
                val_b_loss = val_b.loss, val_b_acc = val_b.accuracy)
            final_row = row
            open(metrics_path, "a") do io
                println(io, join([row.step, row.cycle, row.present_a, row.present_b,
                    row.counts_a, row.counts_b, @sprintf("%.12e", row.loss_ref),
                    @sprintf("%.12e", row.loss_local), @sprintf("%.6e", row.peer_param),
                    @sprintf("%.6e", row.peer_mom1), @sprintf("%.6e", row.peer_mom2),
                    string(row.peer_beta), @sprintf("%.12e", row.val_a_loss),
                    @sprintf("%.12f", row.val_a_acc), @sprintf("%.12e", row.val_b_loss),
                    @sprintf("%.12f", row.val_b_acc)], '\t'))
            end
            if save_trajectory
                snapshots[step] = copy(local_snap.values)
                ref_snapshots[step] = copy(ref_values)
                losses[step] = loss_ref
            end
            if checkpoint_path !== nothing && step == checkpoint_at
                payload = make_payload(step, state, model, ref_model, ref_tree)
                CGCheckpoint.save_checkpoint(checkpoint_path, payload)
                checkpoint_digest = CGCheckpoint.checkpoint_digest(payload)
                println("DEMO_CHECKPOINT step_saved=$step step_next=$(step + 1) " *
                        "path=$checkpoint_path digest=$checkpoint_digest")
            end
        end
        if checkpoint_path !== nothing && step == checkpoint_at
            MPI.Barrier(comm)
        end
    end

    if rank == 0
        CGMetrics.validate_outputs(out, mode, start_step, stop_step, R, sched, dataset;
            run_id, schedule_digest=trajectory_meta.schedule_digest, eta, checkpoint_step)
        println("DEMO_SCHEMA_OK mode=$mode validation_steps=$(stop_step-start_step+2)")
    end

    trajectory_path = nothing
    if rank == 0 && save_trajectory
        trajectory_path = joinpath(out, "trajectory.jls")
        trajectory = (; mode, label, run_id, checkpoint_step = checkpoint_at,
            checkpoint_digest, steps = stop_step, snapshots, ref_snapshots, losses,
            layout = last_layout, tolerances = tol,
            schedule_digest = trajectory_meta.schedule_digest,
            dataset_digest = trajectory_meta.dataset_digest,
            model_definition_version = trajectory_meta.model_definition_version,
            routing = ROUTING_CYCLE)
        open(trajectory_path, "w") do io
            serialize(io, trajectory)
        end
        println("DEMO_TRAJECTORY path=$trajectory_path steps=$(length(snapshots)) " *
                "checkpoint_step=$(checkpoint_at)")
    end

    return (; final_row, trajectory_path, checkpoint_digest)
end

# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

function main_prepare(opts, flags)
    out = require_opt(opts, "out")
    mkpath(out)
    dataset_name = get(opts, "dataset", "toy")
    dataset_name in ("toy", "mnist-fashion") || error("--dataset must be toy or mnist-fashion")
    k2 = dataset_name == "mnist-fashion"
    path = joinpath(out, "$(dataset_name)_prepared.jls")
    isfile(path) && error("refusing to overwrite prepared artifact $path")
    steps = int_opt(opts, "steps", k2 ? 120 : 16)
    steps >= 6 || error("at least six steps required to cover the routing cycle")
    global_batch = int_opt(opts, "global-batch", 128)
    worlds = parse.(Int, split(get(opts, "worlds", "2,4"), ','))
    train_per_task = int_opt(opts, "train-per-task", k2 ? 5120 : CGDataset.TRAIN_PER_TASK)
    val_per_task = int_opt(opts, "val-per-task", k2 ? 1000 : CGDataset.VAL_PER_TASK)
    seed_train = int_opt(opts, "seed-train", 101)
    seed_val = int_opt(opts, "seed-val", 202)
    seed_init = int_opt(opts, "seed-init", 303)
    seed_order = int_opt(opts, "seed-order", 404)
    # The Phase K prompt recommends eta=1e-3. The 16-update toy budget starts
    # head B on an inverted random boundary for the prepared seed, and 1e-3
    # leaves it inverted after 16 updates (measured; see the Phase K devlog).
    # 1e-2 demonstrates learning for both heads within the bounded budget.
    eta = float_opt(opts, "eta", k2 ? 1.0e-3 : 1.0e-2)
    isfinite(eta) && eta > 0 || error("eta must be finite and positive")
    seed_subset = int_opt(opts, "seed-subset", 505)

    dataset = k2 ? CGDataset.make_mnist_fashion_dataset(get(opts, "cache", joinpath(out, "raw"));
        train_per_task, val_per_task, seed_subset) : CGDataset.make_toy_dataset(; seed_train, seed_val,
        train_per_task, val_per_task)
    model = k2 ? Flux.f32(CGModel.build_toy_model(seed_init; indim=784, hidden=32, nclasses=10)) :
        Flux.f64(CGModel.build_toy_model(seed_init))
    CGCommon.ensure(model.head_a.weight != model.head_b.weight,
        "head_a and head_b must have distinct initial parameter values")
    tree = Optimisers.setup(Optimisers.Adam(eta), model)

    schedules = Dict{Int,CGSchedule.Schedule}()
    for R in worlds
        global_batch % R == 0 || error("global batch $global_batch is not divisible " *
                                       "by world size $R")
        local_batch = global_batch ÷ R
        train_per_task >= global_batch || error("task pools must hold a full global batch")
        schedule = CGSchedule.build_schedule(; steps, workers = R, local_batch,
            pool_sizes = (; a = train_per_task, b = train_per_task),
            order_seed = seed_order)
        CGSchedule.validate_schedule(schedule)
        schedules[R] = schedule
    end

    payload = (; schema_version = 2, kind = :phase_k_prepared,
        created_at = string(Dates.now()),
        host = CGCommon.host_name(), julia = string(VERSION),
        dataset, dataset_digest = dataset.digest,
        model, tree,
        model_definition_version = CGModel.MODEL_DEFINITION_VERSION,
        optimizer = (; name = "Adam", eta, beta = (0.9, 0.999), epsilon = 1.0e-8),
        schedules, default_steps = steps, global_batch,
        routing = ROUTING_CYCLE,
        seeds = (; train = seed_train, val = seed_val, init = seed_init, order = seed_order, subset=seed_subset),
        package_versions = CGCommon.package_versions(),
        flux_path = pathof(Flux),
        flux_head = CGCommon.git_head(normpath(joinpath(dirname(pathof(Flux)), ".."))),
        source_hashes = demo_source_hashes())

    open(path, "w") do io
        serialize(io, payload)
    end
    CGDataset.write_provenance(out, dataset)
    for (R, schedule) in schedules
        open(joinpath(out, "schedule-ranks$R.tsv"), "w") do io
            println(io, "step\trank\ttask\thead\tpool_indices\tofficial_indices")
            for step in 1:steps, rank in 0:R-1
                task = schedule.heads[step, rank+1]
                ids = schedule.samples[step, rank+1]
                pool = dataset.pools[task].train
                official = hasproperty(pool, :indices) ? pool.indices[ids] : ids
                println(io, join((step, rank, task, task, join(ids, ','), join(official, ',')), '\t'))
            end
        end
    end

    summary = IOBuffer()
    println(summary, "prepared\t", path)
    println(summary, "dataset_digest\t", dataset.digest)
    if hasproperty(dataset, :loader)
        println(summary, "loader\t", dataset.loader)
        println(summary, "preprocessing\t", dataset.preprocessing)
        println(summary, "label_encoding\t", dataset.label_encoding)
        println(summary, "task_mapping\t", dataset.task_mapping)
    end
    println(summary, "seeds\t", payload.seeds)
    println(summary, "command\t", Base.julia_cmd(), " ", join(_RAW_ARGS, ' '))
    for pkg in payload.package_versions
        println(summary, "package\t$(pkg.name)\t$(pkg.version)\t$(pkg.path)")
    end
    for (file, hash) in sort(collect(payload.source_hashes); by=first)
        println(summary, "source_hash\t$file\t$hash")
    end
    println(summary, "model_definition_version\t", CGModel.MODEL_DEFINITION_VERSION)
    println(summary, "optimizer\tAdam eta=", eta, " beta=(0.9, 0.999)")
    for R in sort(collect(keys(schedules)))
        s = schedules[R]
        println(summary, "schedule_workers\t$R\tdigest=", CGSchedule.schedule_digest(s),
            "\tsteps=", s.steps, "\tlocal_batch=", s.local_batch,
            "\tglobal_batch=", s.global_batch)
        for t in CGSchedule.schedule_summary(s).tasks
            println(summary, "schedule_task\tworkers=$R\ttask=", t.task,
                "\texposure=", t.exposure, "\tpool=", t.pool,
                "\tcompleted_passes=", t.completed_passes, "\tpartial=", t.partial,
                "\treused=", t.reused)
        end
    end
    CGCommon.write_text(joinpath(out, "prepare_metadata.txt"), String(take!(summary)))

    println("DEMO_PREPARE_DONE path=$path dataset_digest=$(dataset.digest) " *
            "worlds=$(sort(collect(keys(schedules)))) model_version=$(CGModel.MODEL_DEFINITION_VERSION)")
    return nothing
end

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

function main_train(opts, flags)
    prepared_path = require_opt(opts, "prepared")
    out = require_opt(opts, "out")
    backend_name = get(opts, "backend", "mpi")
    label = get(opts, "label", "toy-train")
    save_at = int_opt(opts, "save-at", 2)
    save_trajectory = "save-trajectory" in flags

    isfile(prepared_path) || error("prepared artifact $prepared_path does not exist")
    prepared = open(deserialize, prepared_path)
    prepared.kind in (:phase_k_prepared, :phase_k_toy_prepared) ||
        error("$prepared_path is not a Phase K prepared artifact")
    dtype = is_on_gpu() ? Float32 : eltype(prepared.model.trunk.weight)
    TARGET_DTYPE[] = dtype
    tol = resolve_tolerances(opts, dtype)
    CGCommon.ensure(CGDataset.dataset_digest(prepared.dataset.pools) == prepared.dataset_digest,
        "prepared dataset digest mismatch")

    ctx = init_backend(backend_name)
    (; backend, rec, rank, workers, comm, on_gpu) = ctx

    schedule = get(prepared.schedules, workers, nothing)
    schedule === nothing && error("prepared artifact has no schedule for world size " *
                                  "$workers (available: $(sort(collect(keys(prepared.schedules)))))")
    CGSchedule.validate_schedule(schedule)
    CGCommon.ensure(workers >= 2, "the conditional demo needs at least 2 ranks")

    model = to_target_model(prepared.model)
    tree = to_target_tree(prepared.tree)
    ref_model = copy_model(model)
    ref_tree = to_target_tree(prepared.tree)
    state = W(rec, tree)

    if rank == 0
        start_local = CGCommon.snapshot(model, tree).values
        start_ref = CGCommon.snapshot(ref_model, ref_tree).values
        CGCommon.ensure(start_local == start_ref,
            "distributed and reference models do not start from identical values")
    end

    out_abs = abspath(out)
    isfile(joinpath(out_abs, "metadata.txt")) && error("refusing existing run output $out_abs")
    run_id = get(opts, "run-id", basename(dirname(out_abs)) * "-" * basename(out_abs))
    save_at == 2 || error("Phase K continuation requires --save-at 2")
    rank == 0 && mkpath(out_abs)
    MPI.Barrier(comm)
    checkpoint_path = save_at > 0 ? joinpath(out_abs, "checkpoint-step$save_at.jls") : nothing

    schedule_digest = CGSchedule.schedule_digest(schedule)
    if rank == 0
        text = metadata_text(; mode = "train", label, backend = backend_name,
            workers, dtype = string(dtype), tol, sched = schedule, dataset = prepared.dataset,
            extra = (; prepared = abspath(prepared_path),
                run_id,
                prepared_sha256=CGCommon.source_hash(prepared_path),
                prepared_dataset_digest = prepared.dataset_digest,
                schedule_digest, model_definition_version = CGModel.MODEL_DEFINITION_VERSION,
                optimizer = "Adam eta=$(prepared.optimizer.eta)",
                routing = "six-step cycle (see schedule module)",
                on_gpu = on_gpu))
        CGCommon.write_text(joinpath(out_abs, "metadata.txt"), text)
        println("DEMO_META mode=train rank=0 backend=$backend_name ranks=$workers " *
                "dtype=$dtype tolerances=\"$tol\" flux=$(pathof(Flux)) " *
                "schedule_digest=$schedule_digest")
    end

    make_payload = (step, state, model, ref_model, ref_tree) -> begin
        save_state = copy_model(model)
        save_ref = copy_model(ref_model)
        (; schema_version = CGCheckpoint.SCHEMA_VERSION,
            kind = :phase_k_checkpoint,
            created_at = string(Dates.now()),
            label, run_id,
            step_saved = step,
            step_next = step + 1,
            world_size = workers,
            backend_name = String(backend_name),
            dtype = string(dtype),
            julia_version = string(VERSION),
            tolerances = tol,
            model_definition_version = CGModel.MODEL_DEFINITION_VERSION,
            optimizer = prepared.optimizer,
            schedule,
            schedule_digest,
            dataset = prepared.dataset,
            dataset_digest = prepared.dataset_digest,
            model = hostify_model(save_state),
            tree = hostify_tree(state.tree),
            ref_model = hostify_model(save_ref),
            ref_tree = hostify_tree(ref_tree),
            rng = nothing,
            cursors = (; a=sum(schedule.task_requests[1:step, 1]), b=sum(schedule.task_requests[1:step, 2])),
            package_versions = CGCommon.package_versions(),
            flux_path = pathof(Flux),
            flux_head = CGCommon.git_head(normpath(joinpath(dirname(pathof(Flux)), ".."))),
            source_hashes = demo_source_hashes())
    end

    result = run_steps(; mode = "train", label, rank, R = workers, on_gpu, backend, rec,
        comm, tol, sched = schedule, dataset = prepared.dataset, model, state,
        ref_model, ref_tree, out = out_abs,
        start_step = 1, stop_step = schedule.steps,
        checkpoint_at = save_at, checkpoint_path,
        make_payload, save_trajectory, run_id,
        expected_values = nothing,
        trajectory_meta = (; schedule_digest, dataset_digest = prepared.dataset_digest,
            model_definition_version = CGModel.MODEL_DEFINITION_VERSION))

    println(@sprintf("DEMO_RANK_DONE mode=train rank=%d steps=%d exit=ok", rank,
        schedule.steps))
    MPI.Barrier(comm)
    if rank == 0
        println("DEMO_TRAIN_DONE mode=train backend=$backend_name ranks=$workers " *
                "steps=$(schedule.steps) checkpoint=$(checkpoint_path === nothing ? "none" : checkpoint_path) " *
                "trajectory=$(result.trajectory_path === nothing ? "none" : result.trajectory_path)")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

function main_resume(opts, flags)
    checkpoint_path = require_opt(opts, "checkpoint")
    expected_path = require_opt(opts, "expected")
    out = require_opt(opts, "out")
    backend_name = get(opts, "backend", "mpi")
    label = get(opts, "label", "toy-resume")

    payload = CGCheckpoint.load_checkpoint(checkpoint_path)
    payload.kind in (:phase_k_checkpoint, :phase_k_toy_checkpoint) ||
        error("$checkpoint_path is not a Phase K toy checkpoint")
    dtype = is_on_gpu() ? Float32 : eltype(payload.model.trunk.weight)
    TARGET_DTYPE[] = dtype
    tol = resolve_tolerances(opts, dtype)
    run_id = payload.run_id
    CGCommon.ensure(payload.package_versions == CGCommon.package_versions(), "checkpoint package versions/paths changed")
    CGCommon.ensure(payload.source_hashes == demo_source_hashes(), "checkpoint source hashes changed")
    CGCommon.ensure(payload.julia_version == string(VERSION), "checkpoint Julia version changed")
    CGCommon.ensure(payload.tolerances == tol, "checkpoint comparison tolerances changed")
    CGCommon.ensure(CGDataset.dataset_digest(payload.dataset.pools) == payload.dataset_digest,
        "checkpoint dataset digest mismatch")
    expected = open(deserialize, expected_path)
    CGCommon.ensure(expected.run_id == run_id, "trajectory run identity mismatch")

    ctx = init_backend(backend_name)
    (; backend, rec, rank, workers, comm, on_gpu) = ctx

    CGCommon.ensure(payload.world_size == workers,
        "checkpoint was saved for $(payload.world_size) ranks, running $workers")
    CGCommon.ensure(payload.backend_name == backend_name,
        "checkpoint backend $(payload.backend_name) != run backend $backend_name")
    CGCommon.ensure(payload.dtype == string(dtype),
        "checkpoint dtype $(payload.dtype) != run dtype $dtype")
    CGCommon.ensure(payload.model_definition_version == CGModel.MODEL_DEFINITION_VERSION,
        "checkpoint model definition $(payload.model_definition_version) != " *
        "$(CGModel.MODEL_DEFINITION_VERSION)")

    schedule = payload.schedule
    CGSchedule.validate_schedule(schedule)
    CGCommon.ensure(CGSchedule.schedule_digest(schedule) == payload.schedule_digest,
        "checkpoint schedule does not match its own digest")
    CGCommon.ensure(expected.schedule_digest == payload.schedule_digest,
        "saved trajectory is for a different schedule")
    CGCommon.ensure(expected.dataset_digest == payload.dataset_digest,
        "saved trajectory is for a different dataset")
    CGCommon.ensure(expected.checkpoint_step == payload.step_saved,
        "saved trajectory checkpoint step $(expected.checkpoint_step) != " *
        "$(payload.step_saved)")
    CGCommon.ensure(payload.step_next == payload.step_saved + 1,
        "checkpoint step_next $(payload.step_next) != step_saved + 1")
    CGCommon.ensure(expected.checkpoint_digest == CGCheckpoint.checkpoint_digest(payload),
        "checkpoint payload does not match the one the trajectory was saved from")

    model = to_target_model(payload.model)
    tree = to_target_tree(payload.tree)
    ref_model = to_target_model(payload.ref_model)
    ref_tree = to_target_tree(payload.ref_tree)
    state = W(rec, tree)

    # Assert the restored state before the first resumed update: values,
    # moments, rules, and frozen flags, on every rank.
    loaded = CGCommon.snapshot(model, tree)
    saved = CGCommon.snapshot(payload.model, payload.tree)
    CGCommon.ensure(loaded.values == saved.values,
        "rank $rank: loaded model/tree values differ from the checkpoint payload")
    CGCommon.check_leaf_config(payload.tree, tree; context = "rank $rank restored tree")
    ref_loaded = CGCommon.snapshot(ref_model, ref_tree)
    ref_saved = CGCommon.snapshot(payload.ref_model, payload.ref_tree)
    CGCommon.ensure(ref_loaded.values == ref_saved.values,
        "rank $rank: loaded reference state differs from the checkpoint payload")
    CGCommon.check_leaf_config(payload.ref_tree, ref_tree; context="restored reference")
    CGCommon.ensure(loaded.values == expected.snapshots[payload.step_saved],
        "loaded DDP checkpoint does not match saved uninterrupted step")
    CGCommon.ensure(ref_loaded.values == expected.ref_snapshots[payload.step_saved],
        "loaded reference checkpoint does not match saved uninterrupted step")

    out_abs = abspath(out)
    isfile(joinpath(out_abs, "metadata.txt")) && error("refusing existing resume output $out_abs")
    rank == 0 && mkpath(out_abs)
    MPI.Barrier(comm)
    if rank == 0
        text = metadata_text(; mode = "resume", label, backend = backend_name,
            workers, dtype = string(dtype), tol, sched = schedule, dataset = payload.dataset,
            extra = (; checkpoint = abspath(checkpoint_path),
                run_id,
                expected = abspath(expected_path),
                checkpoint_digest = CGCheckpoint.checkpoint_digest(payload),
                step_next = payload.step_next,
                schedule_digest = payload.schedule_digest,
                restored_rules_and_frozen = true,
                on_gpu = on_gpu))
        CGCommon.write_text(joinpath(out_abs, "metadata.txt"), text)
        println("DEMO_META mode=resume rank=0 backend=$backend_name ranks=$workers " *
                "dtype=$dtype tolerances=\"$tol\" flux=$(pathof(Flux)) " *
                "step_next=$(payload.step_next)")
    end

    result = run_steps(; mode = "resume", label, rank, R = workers, on_gpu, backend, rec,
        comm, tol, sched = schedule, dataset = payload.dataset, model, state,
        ref_model, ref_tree, out = out_abs,
        start_step = payload.step_next, stop_step = schedule.steps,
        checkpoint_at = 0, checkpoint_path = nothing, make_payload = nothing,
        save_trajectory = false, expected_values = expected.snapshots, run_id,
        expected_reference = expected.ref_snapshots, checkpoint_step=payload.step_saved,
        trajectory_meta = (; schedule_digest = payload.schedule_digest,
            dataset_digest = payload.dataset_digest,
            model_definition_version = payload.model_definition_version))

    println(@sprintf("DEMO_RANK_DONE mode=resume rank=%d start_step=%d steps=%d exit=ok",
        rank, payload.step_next, schedule.steps))
    MPI.Barrier(comm)
    if rank == 0
        println("DEMO_RESUME_DONE mode=resume backend=$backend_name ranks=$workers " *
                "start_step=$(payload.step_next) stop_step=$(schedule.steps)")
    end
    return nothing
end

# ---------------------------------------------------------------------------

function main()
    opts, flags = parse_options(_RAW_ARGS)
    expected_flux = get(opts, "flux-root", normpath(joinpath(@__DIR__, "../../../flux-integration")))
    realpath(dirname(dirname(pathof(Flux)))) == realpath(expected_flux) ||
        error("Flux resolves to $(pathof(Flux)), expected $expected_flux; use the implementation test environment")
    if MODE == "prepare"
        main_prepare(opts, flags)
    elseif MODE == "train"
        main_train(opts, flags)
    elseif MODE == "resume"
        main_resume(opts, flags)
    else
        error("unknown mode $(repr(MODE)); expected prepare, train, or resume")
    end
    return nothing
end

main()

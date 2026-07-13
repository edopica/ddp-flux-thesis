#!/usr/bin/env julia
"""
verify_gradients.jl — C5 & C6: Multi-step gradient synchronization verifier

Loads the deterministic single-process baseline, sets up the DistributedDataContainer
and DistributedOptimizer, and runs the same 20 steps using distributed data parallelism.
At each step, it verifies that the DDP-averaged gradients and the model parameters
match the global-batch single-process baseline up to numerical tolerance.
"""

using MPI
using Flux
using Flux: DistributedUtils
using Zygote
using Optimisers
using Functors

using MLUtils

include(joinpath(@__DIR__, "..", "reference", "ReferenceLoop.jl"))
using .ReferenceLoop

function collect_arrays(x)
    arrays = AbstractArray[]
    Functors.fmap(x; exclude=a -> a isa AbstractArray) do a
        push!(arrays, a)
        a
    end
    return arrays
end

function compare_structures(label::String, expected, actual, step::Int, tol::Float64=1e-5)
    exp_arrays = collect_arrays(expected)
    act_arrays = collect_arrays(actual)

    if length(exp_arrays) != length(act_arrays)
        return (false, "Step $step $label: array count mismatch (expected=$(length(exp_arrays)), actual=$(length(act_arrays)))")
    end

    for (i, (e, a)) in enumerate(zip(exp_arrays, act_arrays))
        if size(e) != size(a)
            return (false, "Step $step $label array $i: size mismatch (expected=$(size(e)), actual=$(size(a)))")
        end
        diff = maximum(abs.(e .- a))
        if diff > tol
            return (false, "Step $step $label array $i: max difference $diff exceeds tolerance $tol")
        end
    end

    return (true, "")
end

function main()
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
    
    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    rank == 0 && println("=" ^ 65)
    rank == 0 && println("C5 & C6: Multi-step DDP Gradient Verification (Ranks: $world)")
    rank == 0 && println("=" ^ 65)

    baseline_path = joinpath(@__DIR__, "..", "..", "artifacts", "baselines", "reference_loop_baseline.jld2")
    if !isfile(baseline_path)
        rank == 0 && println("FAIL: Baseline file not found at $baseline_path")
        exit(1)
    end
    
    baseline = ReferenceLoop.load_baseline(baseline_path)
    config = baseline.config
    
    # Broadcast initial model and state to ensure perfect alignment
    model = deepcopy(baseline.initial_model)
    model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
    
    rule = Descent(0.01f0)
    dopt = DistributedUtils.DistributedOptimizer(backend, rule)
    st_opt = Optimisers.setup(dopt, model)
    st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0)

    if rank == 0
        if collect_arrays(baseline.initial_state) == collect_arrays(st_opt)
            println("Initial states match? true")
        else
            println("Initial states match? false")
            println("baseline state: ", typeof(baseline.initial_state))
            println("st_opt state: ", typeof(st_opt))
        end
    end
    
    # Dataset partitioning
    (x, y) = config.data_fn()
    ddc = DistributedUtils.DistributedDataContainer(backend, (x, y))
    
    # Get local batch (all of this rank's partition)
    # Using length(ddc) gives the local worker's partition size
    local_data = MLUtils.getobs(ddc, 1:length(ddc))
    local_x, local_y = local_data
    
    num_steps = config.num_steps
    pass = true
    first_failure = ""
    
    for step in 1:num_steps
        b = baseline.steps[step]
        
        # Local forward and backward
        l, grad = Zygote.withgradient(config.loss_fn, model, local_x, local_y)
        
        # DDP implicit averaging + parameter update
        # DistributedOptimizer averages grad[1] IN-PLACE!
        st_opt, model = Optimisers.update(st_opt, model, grad[1])
        
        # Check averaged gradients
        if b.gradients !== nothing
            ok, msg = compare_structures("gradients vs baseline", b.gradients, grad[1], step, 1e-5)
            if !ok
                if pass first_failure = msg end
                pass = false
                rank == 0 && println("  ❌ $msg")
                break
            end
        end
        
        # Check updated model parameters against baseline
        if b.parameters !== nothing
            ok, msg = compare_structures("parameters vs baseline", b.parameters, model, step, 1e-5)
            if !ok
                if pass first_failure = msg end
                pass = false
                rank == 0 && println("  ❌ $msg")
                break
            end
        end
        
        # C6: Verify parameters remain synchronized ACROSS RANKS
        # Rank 0 broadcasts its parameters, and we check if local parameters match exactly
        model_rank0 = deepcopy(model)
        model_rank0 = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model_rank0); root=0)
        ok_sync, msg_sync = compare_structures("cross-rank synchronization", model_rank0, model, step, 0.0)
        if !ok_sync
            if pass first_failure = msg_sync end
            pass = false
            println("Rank $rank ❌ $msg_sync")
            break
        end
        
        # For loss, the sum of local losses / world_size does not necessarily exactly equal the global loss mathematically 
        # (if sizes differ or depending on the reduction). But for equally split datasets with MSE, 
        # the average of local MSEs is exactly global MSE.
        # Since local ranks don't compute average loss here, we don't strictly compare `l` across all ranks,
        # but we could. For C5, gradient and parameter matching is the core deliverable.
    end
    
    if rank == 0
        println("\n" * "=" ^ 65)
        if pass
            println("✅ C5 & C6 VERIFICATION PASSED: All $num_steps steps match global baseline")
        else
            println("❌ C5 & C6 VERIFICATION FAILED: $first_failure")
        end
        println("=" ^ 65)
    end
    
    MPI.Barrier(backend.comm)
    pass || exit(1)
end

main()

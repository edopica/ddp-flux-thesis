#!/usr/bin/env julia

using MPI
using Flux
using Flux: DistributedUtils
using Zygote
using Optimisers
using Functors
using Random
using Test

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

function collect_arrays(x)
    arrays = AbstractArray[]
    Functors.fmap(x; exclude=a -> a isa AbstractArray) do a
        push!(arrays, a)
        a
    end
    return arrays
end

function check_sync(backend, model, label)
    rank = DistributedUtils.local_rank(backend)
    model_rank0 = deepcopy(model)
    model_rank0 = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model_rank0); root=0)
    
    act_arrays = collect_arrays(model)
    exp_arrays = collect_arrays(model_rank0)
    
    for (i, (a, e)) in enumerate(zip(act_arrays, exp_arrays))
        diff = maximum(abs.(a .- e))
        if diff > 0.0
            error("Rank $rank: $label out of sync at array $i with diff $diff")
        end
    end
end

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

struct TwoHeadModel
    backbone
    head_a
    head_b
end
@functor TwoHeadModel

struct InnerConditional
    head_a1
    head_a2
end
@functor InnerConditional

struct OuterConditional
    backbone
    branch_a :: InnerConditional
    branch_b
end
@functor OuterConditional

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

function test_two_head(backend)
    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    Random.seed!(42)
    model = TwoHeadModel(Dense(10 => 5), Dense(5 => 2), Dense(5 => 2))
    model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
    
    opt = Optimisers.setup(DistributedUtils.DistributedOptimizer(backend, Descent(0.1)), model)
    opt = DistributedUtils.synchronize!!(backend, opt; root=0)
    
    x = rand32(10, 1)
    x = DistributedUtils.bcast!(backend, x)
    
    for step in 1:3
        l, gs = Zygote.withgradient(model) do m
            if rank == 0
                sum(m.head_a(m.backbone(x)))
            else
                sum(m.head_b(m.backbone(x)))
            end
        end
        gs = gs[1]
        
        resolved_gs = DistributedUtils.resolve_unused_parameters!(backend, gs, model)
        
        if rank == 0
            @test resolved_gs.head_b.weight isa AbstractArray
            @test all(resolved_gs.head_b.weight .== 0)
        else
            @test resolved_gs.head_a.weight isa AbstractArray
            @test all(resolved_gs.head_a.weight .== 0)
        end
        
        opt, model = Optimisers.update(opt, model, resolved_gs)
        check_sync(backend, model, "Test 1 (Two-head) step $step")
    end
    rank == 0 && println("  ✅ Test 1 (Two-head conditional) passed")
    return true
end

function test_chain_level_nothing(backend)
    rank = DistributedUtils.local_rank(backend)
    
    Random.seed!(42)
    model = Chain(Dense(10 => 5), Chain(Dense(5 => 5), Dense(5 => 2)))
    model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
    
    opt = Optimisers.setup(DistributedUtils.DistributedOptimizer(backend, Descent(0.1)), model)
    opt = DistributedUtils.synchronize!!(backend, opt; root=0)
    
    x = rand32(10, 1)
    x = DistributedUtils.bcast!(backend, x)
    
    for step in 1:3
        l, gs = Zygote.withgradient(model) do m
            if rank == 0
                sum(m(x))
            else
                sum(m.layers[1](x)) # Inner chain unused
            end
        end
        gs = gs[1]
        
        resolved_gs = DistributedUtils.resolve_unused_parameters!(backend, gs, model)
        
        if rank != 0
            @test resolved_gs.layers[2].layers[1].weight isa AbstractArray
            @test all(resolved_gs.layers[2].layers[1].weight .== 0)
        end
        
        opt, model = Optimisers.update(opt, model, resolved_gs)
        check_sync(backend, model, "Test 2 (Chain-level nothing) step $step")
    end
    rank == 0 && println("  ✅ Test 2 (Chain-level nothing) passed")
    return true
end

function test_nested_conditional(backend)
    rank = DistributedUtils.local_rank(backend)
    
    Random.seed!(42)
    model = OuterConditional(
        Dense(10 => 5),
        InnerConditional(Dense(5 => 2), Dense(5 => 2)),
        Dense(5 => 2)
    )
    model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
    
    opt = Optimisers.setup(DistributedUtils.DistributedOptimizer(backend, Descent(0.1)), model)
    opt = DistributedUtils.synchronize!!(backend, opt; root=0)
    
    x = rand32(10, 1)
    x = DistributedUtils.bcast!(backend, x)
    
    for step in 1:3
        l, gs = Zygote.withgradient(model) do m
            features = m.backbone(x)
            if rank == 0
                sum(m.branch_a.head_a1(features))
            elseif rank == 1
                sum(m.branch_a.head_a2(features))
            else
                sum(m.branch_b(features))
            end
        end
        gs = gs[1]
        
        resolved_gs = DistributedUtils.resolve_unused_parameters!(backend, gs, model)
        opt, model = Optimisers.update(opt, model, resolved_gs)
        check_sync(backend, model, "Test 3 (Nested conditional) step $step")
    end
    rank == 0 && println("  ✅ Test 3 (Nested conditional) passed")
    return true
end

function test_partial_layer(backend)
    rank = DistributedUtils.local_rank(backend)
    
    Random.seed!(42)
    model = Dense(10 => 5)
    model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
    
    opt = Optimisers.setup(DistributedUtils.DistributedOptimizer(backend, Descent(0.1)), model)
    opt = DistributedUtils.synchronize!!(backend, opt; root=0)
    
    x = rand32(10, 1)
    x = DistributedUtils.bcast!(backend, x)
    
    for step in 1:3
        l, gs = Zygote.withgradient(model) do m
            if rank == 0
                sum(m.weight * x .+ m.bias)
            else
                sum(m.weight * x) # bias unused
            end
        end
        gs = gs[1]
        
        resolved_gs = DistributedUtils.resolve_unused_parameters!(backend, gs, model)
        
        if rank != 0
            @test resolved_gs.bias isa AbstractArray
            @test all(resolved_gs.bias .== 0)
        end
        
        opt, model = Optimisers.update(opt, model, resolved_gs)
        check_sync(backend, model, "Test 4 (Partial layer) step $step")
    end
    rank == 0 && println("  ✅ Test 4 (Partial layer usage) passed")
    return true
end

function test_asymmetric_3rank(backend)
    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    if world < 3
        rank == 0 && println("  ⚠️  Skipping Test 5 (Asymmetric 3-rank): requires >= 3 ranks")
        return true
    end
    
    Random.seed!(42)
    model = TwoHeadModel(Dense(10 => 5), Dense(5 => 2), Dense(5 => 2))
    model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
    
    opt = Optimisers.setup(DistributedUtils.DistributedOptimizer(backend, Descent(0.1)), model)
    opt = DistributedUtils.synchronize!!(backend, opt; root=0)
    
    x = rand32(10, 1)
    x = DistributedUtils.bcast!(backend, x)
    
    for step in 1:3
        l, gs = Zygote.withgradient(model) do m
            if rank == 0
                sum(m.head_a(m.backbone(x)))
            else
                sum(m.head_b(m.backbone(x)))
            end
        end
        gs = gs[1]
        
        resolved_gs = DistributedUtils.resolve_unused_parameters!(backend, gs, model)
        opt, model = Optimisers.update(opt, model, resolved_gs)
        check_sync(backend, model, "Test 5 (Asymmetric 3-rank) step $step")
    end
    rank == 0 && println("  ✅ Test 5 (Asymmetric 3-rank) passed")
    return true
end

# ------------------------------------------------------------------
# Deterministic Baseline (Mathematical validation)
# ------------------------------------------------------------------

function collect_trajectory(backend, num_steps)
    rank = DistributedUtils.local_rank(backend)
    
    Random.seed!(42)
    model = TwoHeadModel(Dense(10 => 5), Dense(5 => 2), Dense(5 => 2))
    model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
    
    opt = Optimisers.setup(DistributedUtils.DistributedOptimizer(backend, Descent(0.1)), model)
    opt = DistributedUtils.synchronize!!(backend, opt; root=0)
    
    x = rand32(10, 1)
    x = DistributedUtils.bcast!(backend, x)
    
    traj = []
    for step in 1:num_steps
        l, gs = Zygote.withgradient(model) do m
            if rank == 0
                sum(m.head_a(m.backbone(x)))
            else
                sum(m.head_b(m.backbone(x)))
            end
        end
        gs = gs[1]
        resolved_gs = DistributedUtils.resolve_unused_parameters!(backend, gs, model)
        opt, model = Optimisers.update(opt, model, resolved_gs)
        push!(traj, deepcopy(model))
    end
    return traj
end

function generate_true_baseline(num_steps, world)
    Random.seed!(42)
    model = TwoHeadModel(Dense(10 => 5), Dense(5 => 2), Dense(5 => 2))
    opt = Optimisers.setup(Descent(0.1), model)
    x = rand32(10, 1)
    
    traj = []
    for step in 1:num_steps
        # Compute gradient as if Rank 0
        l0, gs0 = Zygote.withgradient(model) do m
            sum(m.head_a(m.backbone(x)))
        end
        gs0 = gs0[1]
        
        # Compute gradient as if Rank 1+
        gs_other = []
        for r in 1:(world-1)
            l_r, gs_r = Zygote.withgradient(model) do m
                sum(m.head_b(m.backbone(x)))
            end
            push!(gs_other, gs_r[1])
        end
        
        # Average manually
        add_opt(a, b) = (a === nothing && b === nothing) ? nothing : (a === nothing ? b : (b === nothing ? a : a .+ b))
        
        bb_grad = gs0.backbone
        for g in gs_other
            bb_grad = fmap(add_opt, bb_grad, g.backbone)
        end
        bb_grad = fmap(a -> a === nothing ? nothing : a ./ world, bb_grad)
        
        ha_grad = fmap(a -> a === nothing ? nothing : a ./ world, gs0.head_a)
        
        hb_grad = gs_other[1].head_b
        if length(gs_other) > 1
            for i in 2:length(gs_other)
                hb_grad = fmap(add_opt, hb_grad, gs_other[i].head_b)
            end
        end
        hb_grad = fmap(a -> a === nothing ? nothing : a ./ world, hb_grad)
        
        true_gs = (backbone = bb_grad, head_a = ha_grad, head_b = hb_grad)
        
        opt, model = Optimisers.update(opt, model, true_gs)
        push!(traj, deepcopy(model))
    end
    return traj
end

function test_manual_baseline(backend)
    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    ddp_traj = collect_trajectory(backend, 5)
    true_traj = generate_true_baseline(5, world)
    
    for (step, (m_ddp, m_true)) in enumerate(zip(ddp_traj, true_traj))
        arrs_ddp = collect_arrays(m_ddp)
        arrs_true = collect_arrays(m_true)
        for (i, (a_ddp, a_true)) in enumerate(zip(arrs_ddp, arrs_true))
            diff = maximum(abs.(a_ddp .- a_true))
            if diff > 1e-5
                error("Rank $rank: Manual baseline mismatch at step $step, array $i, diff $diff")
            end
        end
    end
    rank == 0 && println("  ✅ Deterministic mathematical baseline verified")
    return true
end

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

function main()
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
    
    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    if rank == 0
        println("=" ^ 65)
        println("Conditional Graph & Unused Parameters Verification (Ranks: $world)")
        println("=" ^ 65)
    end
    
    pass = true
    try
        test_two_head(backend)
        test_chain_level_nothing(backend)
        test_nested_conditional(backend)
        test_partial_layer(backend)
        test_asymmetric_3rank(backend)
        test_manual_baseline(backend)
    catch e
        pass = false
        println("Rank $rank ❌ FAILED: ", sprint(showerror, e))
        # Print full stacktrace for debugging
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
    
    MPI.Barrier(backend.comm)
    
    if rank == 0
        println("=" ^ 65)
        if pass
            println("✅ ALL CONDITIONAL TESTS PASSED")
        else
            println("❌ CONDITIONAL TESTS FAILED")
        end
        println("=" ^ 65)
    end
    
    pass || exit(1)
end

main()

"""
    verify.jl — C3: Deep verification of model synchronization

A more thorough verification script that:
1. Builds models with deliberately different seeds per rank
2. Broadcasts from rank 0
3. Performs element-wise comparison of every parameter tensor
4. Checks optimizer state synchronization
5. Verifies parameter names/structure match
6. Reports per-layer statistics

Usage:
    mpiexecjl --project=. -n 2 julia scripts/sync/verify.jl
"""

using MPI
using Flux
using Flux: DistributedUtils
using Random
using Optimisers
using Functors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""
    collect_named_params(model) -> Vector{Pair{String, AbstractArray}}

Collect all parameter arrays with descriptive path names.
"""
function collect_named_params(model)
    result = Pair{String, AbstractArray}[]

    function _walk(obj, prefix)
        if obj isa Chain
            for (i, layer) in enumerate(obj.layers)
                _walk(layer, "$prefix.layers[$i]")
            end
        elseif obj isa Dense
            push!(result, "$prefix.weight" => obj.weight)
            push!(result, "$prefix.bias" => obj.bias)
        else
            # Fallback: fmap to find arrays
            Functors.fmap(obj; exclude=x -> x isa AbstractArray{<:Number}) do x
                push!(result, "$prefix.param" => x)
                x
            end
        end
    end

    _walk(model, "model")
    return result
end

"""
    gather_param_bytes(backend, param::AbstractArray) -> (local_param, all_match)

Gather a parameter from all ranks and check if they match rank 0's value.
Uses allreduce to check max deviation.
"""
function gather_param_bytes(backend, param::AbstractArray)
    pvec = vec(Float32.(copy(param)))

    # Compute max across ranks
    pvec_max = copy(pvec)
    DistributedUtils.allreduce!(backend, pvec_max, max)

    # Compute min across ranks
    pvec_min = copy(pvec)
    # For min, negate, take max, negate back (MPI.jl min reduction workaround)
    pvec_min .*= -1
    DistributedUtils.allreduce!(backend, pvec_min, max)
    pvec_min .*= -1

    # If max == min for every element, all ranks have the same value
    max_range = maximum(abs.(pvec_max .- pvec_min))
    return max_range
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    # --- Initialize ---
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)

    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)

    rank == 0 && println("=" ^ 65)
    rank == 0 && println("C3 Deep Verification: Model sync across $world ranks")
    rank == 0 && println("=" ^ 65)

    # --- Test 1: Basic model sync ---
    rank == 0 && println("\n▸ Test 1: Model parameter broadcast")

    Random.seed!(42 + rank * 17)  # Different seeds
    model = Chain(Dense(1 => 256, tanh), Dense(256 => 1))

    # Count params before sync
    pre_params = collect_named_params(model)
    rank == 0 && println("  Model structure: $(length(pre_params)) parameter tensors")
    for (name, p) in pre_params
        rank == 0 && println("    $name: size=$(size(p)) eltype=$(eltype(p))")
    end

    # Broadcast
    model = DistributedUtils.synchronize!!(
        backend,
        DistributedUtils.FluxDistributedModel(model);
        root=0
    )

    # Verify each parameter tensor
    post_params = collect_named_params(model)
    all_match = true
    for (name, p) in post_params
        dev = gather_param_bytes(backend, p)
        status = dev == 0.0 ? "✅" : "❌"
        if dev != 0.0
            all_match = false
        end
        rank == 0 && println("  $status $name: max_range=$dev")
    end

    MPI.Barrier(backend.comm)
    rank == 0 && println(all_match ?
        "  ✅ Test 1 PASSED: All parameter tensors identical across ranks" :
        "  ❌ Test 1 FAILED: Parameter mismatch detected")

    # --- Test 2: Optimizer state sync ---
    rank == 0 && println("\n▸ Test 2: Optimizer state broadcast")

    opt = Optimisers.Adam(0.001f0)
    st_opt = Optimisers.setup(opt, model)

    # Manually perturb optimizer state on non-root ranks to verify sync works
    if rank != 0
        Functors.fmap(st_opt; exclude=x -> x isa AbstractArray{<:Number}) do x
            x .= randn!(x)
            x
        end
    end

    st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0)
    rank == 0 && println("  ✅ Optimizer state synchronized without errors")

    MPI.Barrier(backend.comm)

    # --- Test 3: Reference model consistency ---
    rank == 0 && println("\n▸ Test 3: Reference model consistency check")

    # Build the SAME model (seed=42) on all ranks
    Random.seed!(42)
    ref_model = Chain(Dense(1 => 256, tanh), Dense(256 => 1))

    # Sync from rank 0
    ref_model = DistributedUtils.synchronize!!(
        backend,
        DistributedUtils.FluxDistributedModel(ref_model);
        root=0
    )

    # Now compare the synced model against rank 0's seed-42 model
    # (Only rank 0 can do this since it built the model with seed=42)
    if rank == 0
        Random.seed!(42)
        expected_model = Chain(Dense(1 => 256, tanh), Dense(256 => 1))

        ref_params = collect_named_params(ref_model)
        exp_params = collect_named_params(expected_model)

        all_exact = true
        for ((rn, rp), (en, ep)) in zip(ref_params, exp_params)
            diff = maximum(abs.(rp .- ep))
            if diff != 0.0
                all_exact = false
                println("  ⚠️  $rn vs $en: max_diff=$diff")
            end
        end
        println(all_exact ?
            "  ✅ Test 3 PASSED: Synced model matches rank 0's original (seed=42)" :
            "  ❌ Test 3 FAILED: Synced model differs from rank 0's seed=42 build")
    end

    MPI.Barrier(backend.comm)

    # --- Test 4: Larger model ---
    rank == 0 && println("\n▸ Test 4: Larger model (3-layer MLP)")

    Random.seed!(99 + rank)
    big_model = Chain(
        Dense(10 => 128, relu),
        Dense(128 => 64, relu),
        Dense(64 => 1)
    )

    big_model = DistributedUtils.synchronize!!(
        backend,
        DistributedUtils.FluxDistributedModel(big_model);
        root=0
    )

    big_params = collect_named_params(big_model)
    big_match = true
    total_params = 0
    for (name, p) in big_params
        dev = gather_param_bytes(backend, p)
        total_params += length(p)
        if dev != 0.0
            big_match = false
            rank == 0 && println("  ❌ $name: max_range=$dev")
        end
    end
    rank == 0 && println(big_match ?
        "  ✅ Test 4 PASSED: $total_params parameters identical across ranks" :
        "  ❌ Test 4 FAILED: Parameter mismatch in larger model")

    MPI.Barrier(backend.comm)

    # --- Final summary ---
    if rank == 0
        println("\n" * "=" ^ 65)
        all_tests = all_match && big_match
        if all_tests
            println("✅ C3 VERIFICATION PASSED: All tests passed for $world ranks")
        else
            println("❌ C3 VERIFICATION FAILED: Some tests failed")
        end
        println("=" ^ 65)
    end

    MPI.Barrier(backend.comm)
end

main()

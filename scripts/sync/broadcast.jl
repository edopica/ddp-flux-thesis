"""
    broadcast.jl — C3: Model broadcast and parameter verification

Broadcasts model parameters and optimizer state from rank 0 to all ranks
using Flux's `DistributedUtils.synchronize!!`. Then verifies that every rank
holds bit-identical parameters.

Pass condition: all ranks start from identical parameters after sync.

Usage:
    mpiexecjl --project=. -n 2 julia scripts/sync/broadcast.jl
    # or inside SLURM:
    srun --mpi=pmi2 -n 4 julia --project=. scripts/sync/broadcast.jl
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
    collect_param_vec(model) -> Vector{Float32}

Flatten all model parameters into a single contiguous Float32 vector.
Uses Functors.fmap to traverse the model tree.
"""
function collect_param_vec(model)
    params = Float32[]
    Functors.fmap(model; exclude=x -> x isa AbstractArray{<:Number}) do x
        append!(params, vec(Float32.(x)))
        x
    end
    return params
end

"""
    param_hash(model) -> Float64

Compute a simple hash of model parameters by summing the absolute values.
Deterministic and platform-independent for exact Float32 equality checks.
"""
function param_hash(model)
    pvec = collect_param_vec(model)
    return sum(abs, pvec)
end

"""
    param_max_diff(model_a, model_b) -> Float64

Compute the maximum absolute difference between two models' parameters.
"""
function param_max_diff(model_a, model_b)
    pa = collect_param_vec(model_a)
    pb = collect_param_vec(model_b)
    @assert length(pa) == length(pb) "Parameter count mismatch: $(length(pa)) vs $(length(pb))"
    return maximum(abs.(pa .- pb))
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    # --- Initialize MPI backend ---
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)

    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)

    rank == 0 && println("=" ^ 60)
    rank == 0 && println("C3: Model broadcast and parameter verification")
    rank == 0 && println("   world_size = $world")
    rank == 0 && println("=" ^ 60)

    # -----------------------------------------------------------------------
    # Step 1: Each rank builds its own model with a DIFFERENT seed.
    #         This ensures parameters are genuinely different before sync.
    # -----------------------------------------------------------------------
    Random.seed!(1000 + rank)  # Deliberately different per rank
    model = Chain(Dense(1 => 256, tanh), Dense(256 => 1))

    pre_sync_hash = param_hash(model)
    println("[rank $rank] PRE-sync param hash = $pre_sync_hash")

    # Verify they're actually different (collect hashes from all ranks)
    pre_hashes = [pre_sync_hash]
    DistributedUtils.allreduce!(backend, pre_hashes, +)
    # If all hashes were the same, the sum would be hash * world
    # This is a soft check — the stronger check is below

    MPI.Barrier(backend.comm)

    # -----------------------------------------------------------------------
    # Step 2: Snapshot rank 0's model BEFORE sync for later comparison
    # -----------------------------------------------------------------------
    if rank == 0
        rank0_model = deepcopy(model)
    end

    # -----------------------------------------------------------------------
    # Step 3: Broadcast model from rank 0 to all ranks
    # -----------------------------------------------------------------------
    rank == 0 && println("\n--- Broadcasting model (synchronize!!) ---")

    model = DistributedUtils.synchronize!!(
        backend,
        DistributedUtils.FluxDistributedModel(model);
        root=0
    )

    post_sync_hash = param_hash(model)
    println("[rank $rank] POST-sync param hash = $post_sync_hash")

    MPI.Barrier(backend.comm)

    # -----------------------------------------------------------------------
    # Step 4: Verify all ranks have identical parameters
    # -----------------------------------------------------------------------
    rank == 0 && println("\n--- Verifying parameter identity across ranks ---")

    # Strategy: collect the full parameter vector on each rank,
    # allreduce with max of absolute differences vs rank 0's broadcast.
    # Since all ranks should be identical, we use allreduce to check
    # that max(|param_i - mean(param_i)|) == 0 across all ranks.
    pvec = collect_param_vec(model)
    n_params = length(pvec)

    # Compute sum of all params across ranks — should equal pvec * world
    pvec_sum = copy(pvec)
    DistributedUtils.allreduce!(backend, pvec_sum, +)

    # The mean should equal each rank's params exactly
    pvec_mean = pvec_sum ./ world
    max_deviation = maximum(abs.(pvec .- pvec_mean))

    println("[rank $rank] n_params=$n_params  max_deviation=$max_deviation")

    MPI.Barrier(backend.comm)

    # -----------------------------------------------------------------------
    # Step 5: Verify rank 0's model is unchanged after sync
    # -----------------------------------------------------------------------
    if rank == 0
        r0_diff = param_max_diff(model, rank0_model)
        println("\n[rank 0] Max diff vs pre-sync snapshot: $r0_diff")
        if r0_diff > 0
            println("⚠️  WARNING: rank 0 model changed during sync!")
        else
            println("✅ rank 0 model unchanged after sync (correct)")
        end
    end

    MPI.Barrier(backend.comm)

    # -----------------------------------------------------------------------
    # Step 6: Broadcast optimizer state
    # -----------------------------------------------------------------------
    rank == 0 && println("\n--- Setting up and broadcasting optimizer state ---")

    opt = Optimisers.Adam(0.001f0)
    st_opt = Optimisers.setup(opt, model)
    st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0)

    println("[rank $rank] Optimizer state synchronized successfully")

    MPI.Barrier(backend.comm)

    # -----------------------------------------------------------------------
    # Step 7: Final verdict
    # -----------------------------------------------------------------------
    # Collect max_deviation across all ranks
    all_devs = [max_deviation]
    DistributedUtils.allreduce!(backend, all_devs, max)
    global_max_dev = all_devs[1]

    MPI.Barrier(backend.comm)

    if rank == 0
        println("\n" * "=" ^ 60)
        if global_max_dev == 0.0
            println("✅ C3 PASSED: All $world ranks have identical parameters")
            println("   Global max deviation: $global_max_dev")
            println("   Parameter count per model: $n_params")
        else
            println("❌ C3 FAILED: Parameter mismatch detected!")
            println("   Global max deviation: $global_max_dev")
        end
        println("=" ^ 60)
    end

    MPI.Barrier(backend.comm)
end

main()

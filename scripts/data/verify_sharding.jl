"""
    verify_sharding.jl — C4: Distributed data sharding verification

Verifies that `DistributedDataContainer` pads datasets correctly, ensuring
all MPI ranks process exactly the same number of batches per epoch. This is
required to avoid deadlocks in distributed training loops.

Usage:
    mpiexecjl --project=. -n 4 julia scripts/data/verify_sharding.jl
"""

using MPI
using Flux
using Flux: DistributedUtils
using MLUtils

function main()
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
    
    rank = DistributedUtils.local_rank(backend)
    world_size = DistributedUtils.total_workers(backend)
    
    rank == 0 && println("=" ^ 60)
    rank == 0 && println("C4: Distributed Data Sharding Verification")
    rank == 0 && println("   world_size = $world_size")
    rank == 0 && println("=" ^ 60)

    # Test case 1: N = 11, world_size = 4
    # With ceil(11/4) = 3 per worker, total padded size = 12.
    # Batch size = 2, so each worker should get 2 batches (drop_last=false, ceil(3/2) = 2).
    
    N = 11
    batchsize = 2
    data = 1:N
    
    ddc = DistributedUtils.DistributedDataContainer(backend, data)
    
    # Check length
    expected_length = Int(ceil(N / world_size))
    if length(ddc) != expected_length
        error("Rank $rank: expected DDC length $expected_length, got $(length(ddc))")
    end
    
    loader = DataLoader(ddc, batchsize=batchsize)
    
    # Simulate a training loop
    num_batches = 0
    for batch in loader
        num_batches += 1
        
        # Simulate an Allreduce step
        val = [1.0]
        DistributedUtils.allreduce!(backend, val, +)
    end
    
    println("[rank $rank] processed $num_batches batches.")
    
    # Gather num_batches across all workers
    all_num_batches = [num_batches]
    DistributedUtils.allreduce!(backend, all_num_batches, max)
    max_batches = all_num_batches[1]
    
    all_num_batches_min = [num_batches]
    DistributedUtils.allreduce!(backend, all_num_batches_min, min)
    min_batches = all_num_batches_min[1]
    
    MPI.Barrier(backend.comm)
    
    if rank == 0
        if max_batches == min_batches
            println("✅ C4 PASSED: All ranks processed exactly $max_batches batches.")
        else
            println("❌ C4 FAILED: Ranks processed different number of batches (min: $min_batches, max: $max_batches)")
            exit(1)
        end
        println("=" ^ 60)
    end
end

main()